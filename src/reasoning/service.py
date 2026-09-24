"""The reasoning process: three call sites, one endpoint, one worker (section 5.7).

Wires the Environmental Supervisor, Personal Context and Fault Diagnosis to
the blackboard and runs them. It is a separate process for the reason section
9.3 gives: killing it -- which the demonstration does -- costs the supervisor's
goals, the answers to what an occupant says, and the explanations of faults,
and costs the regulatory loop nothing (FR-11, FR-47).

**Model calls never run on the MQTT thread.** Handlers only record what
arrived; :meth:`ReasoningService.tick` does the work, on the process's own
loop. That is a correctness requirement, not tidiness: the tool client and the
supervisor's verdict wait both listen for answers delivered on the MQTT
thread, and waiting on that thread for its own delivery would never return.

**Order within a tick** puts the cheapest-to-delay last. A newly confirmed
fault is explained first -- the mode has already changed, in the detector
bank's process, and the explanation is what an occupant is now waiting for
(FR-26). Then anything an occupant said. Then, if it is due, the supervisor.
"""

from __future__ import annotations

import logging
import threading
from collections import deque

from src.common import topics
from src.common.clock import Clock
from src.common.config import Config
from src.common.mqtt_client import Blackboard
from src.common.schemas import Utterance
from src.reasoning.audit import ReasoningAudit
from src.reasoning.diagnosis import FaultDiagnoser
from src.reasoning.endpoint import ChatEndpoint
from src.reasoning.single_shot import PersonalContext
from src.reasoning.supervisor_agent import EnvironmentalSupervisor, SupervisorSchedule
from src.reasoning.supervisor_tools import RoomSnapshot, SupervisorTools
from src.reasoning.tool_client import BlackboardToolClient

LOGGER = logging.getLogger(__name__)

#: Most utterances waiting to be answered. Someone talking faster than the
#: model can answer loses the oldest, not the process's memory.
MAX_WAITING_UTTERANCES = 8


class ReasoningService:
    """Everything the reasoning layer does, driven from one loop."""

    def __init__(
        self,
        config: Config,
        clock: Clock,
        blackboard: Blackboard,
        endpoint: ChatEndpoint,
    ) -> None:
        self._config = config
        self._clock = clock
        self._blackboard = blackboard
        audit = ReasoningAudit(clock, blackboard, config.reasoning.max_audit_chars)

        self.snapshot = RoomSnapshot(config, clock, blackboard)
        self._tool_client = BlackboardToolClient(config, clock, blackboard)
        self.supervisor = EnvironmentalSupervisor(
            config,
            clock,
            endpoint,
            SupervisorTools(config, clock, blackboard, self.snapshot),
            audit,
        )
        self.schedule = SupervisorSchedule(config, clock)
        self.diagnoser = FaultDiagnoser(clock, endpoint, audit)
        self.personal_context = PersonalContext(
            config, clock, endpoint, audit, tools=self._tool_client
        )

        self._lock = threading.Lock()
        self._utterances: deque[Utterance] = deque(maxlen=MAX_WAITING_UTTERANCES)

    def subscribe(self) -> None:
        self.snapshot.subscribe()
        self._tool_client.subscribe()
        self._blackboard.subscribe(
            topics.CONTEXT_UTTERANCE, Utterance, self._on_utterance
        )

    def _on_utterance(self, _topic: str, utterance: Utterance) -> None:
        """Record it and return; answering happens on the service loop."""
        with self._lock:
            self._utterances.append(utterance)

    # --- the loop -----------------------------------------------------

    def tick(self) -> None:
        """Do whatever has become due: diagnoses, answers, a supervisor run.

        One failing call site never stops the others. Each is contained here,
        at the process boundary, and logged with its context: a bug in the
        diagnosis prompt must not silence Personal Context.
        """
        self._diagnose_new_faults()
        self._answer_utterances()
        self._supervise_if_due()

    def _diagnose_new_faults(self) -> None:
        for event in self.snapshot.drain_new_faults():
            try:
                diagnosis = self.diagnoser.diagnose(event, self.snapshot.mode)
            except Exception:
                LOGGER.exception("diagnosis of %s failed", event.fault_id)
                continue
            self._blackboard.publish(topics.DIAGNOSIS, diagnosis)
            LOGGER.info("diagnosed %s: %s", event.fault_id, diagnosis.user_message)

    def _answer_utterances(self) -> None:
        with self._lock:
            waiting = list(self._utterances)
            self._utterances.clear()
        for utterance in waiting:
            try:
                hint = self.personal_context.extract(
                    utterance.text, trigger=f"utterance ({utterance.source.value})"
                )
            except Exception:
                LOGGER.exception("answering %r failed", utterance.text)
                continue
            if hint is not None:
                self._blackboard.publish(topics.CONTEXT_PREFERENCE, hint)
                LOGGER.info("answered: %s", hint.spoken_reply or hint.intent.value)

    def _supervise_if_due(self) -> None:
        self.schedule.notice(self.snapshot.drain_events())
        trigger = self.schedule.due()
        if trigger is None:
            return
        try:
            self.supervisor.run(trigger)
        except Exception:
            LOGGER.exception("supervisor run (%s) failed", trigger)


def build_service(
    config: Config, clock: Clock, blackboard: Blackboard, client=None
) -> ReasoningService:
    """Assemble the process. ``client`` replaces the OpenAI client in tests."""
    return ReasoningService(
        config, clock, blackboard, ChatEndpoint(config.reasoning, clock, client)
    )
