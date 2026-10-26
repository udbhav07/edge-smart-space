"""Device services: each actuator beyond the thermal loop, as a service of its own (FR-87).

Layer 1: may import ``src/common`` and nothing else. A device here is reached
only through its topics, so the reasoning layer never knows which of these
exist until their manifests appear on the blackboard.
"""
