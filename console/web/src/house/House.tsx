import { OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Box, Map as MapIcon, MousePointerClick, Rotate3d } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import * as THREE from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { Card, Empty, PageHeader } from "@/components/ui";
import { cn } from "@/lib/format";
import { useConsole } from "@/lib/useConsole";
import { RemoteTile } from "@/remotes/RemoteTile";
import { frameOf } from "./plan";
import { HouseScene } from "./Scene";

// The house, live (FR-99): every device where it stands, doing what its
// service says it is doing. Tap one for its remote; the same remote as on
// the Remotes page, so a tap here goes through the same checks as Jarvis.

type View = "3d" | "top";

export function House() {
  const { state } = useConsole();
  const [view, setView] = useState<View>("3d");
  const [selected, setSelected] = useState<string | null>(null);
  const webgl = useMemo(hasWebGl, []);
  if (!state) return null;
  const plan = state.house;
  const devices = state.devices ?? [];
  const chosen = devices.find((device) => device.device_id === selected) ?? null;
  const roomC = state.room.sensors.temp_01?.value;
  const outsideC = state.room.sensors.outdoor_01?.value;
  const acMode = state.room.actuator?.kind ?? null;
  const thermalRoom = plan?.placements.ac?.room ?? "living_room";
  const robot = devices.find((device) => device.manifest.kind === "robot");
  return (
    <div className="mx-auto max-w-[1500px] px-4 py-6 sm:px-8">
      <PageHeader
        title="House"
        subtitle="Every device in the house, as its service reports it. Switch something on and watch it happen; tap a device for its remote."
        actions={
          <div className="flex rounded-full bg-raised p-1" role="group" aria-label="View">
            {([["3d", "3D", Rotate3d], ["top", "Top view", MapIcon]] as const).map(([key, title, Icon]) => (
              <button
                key={key}
                onClick={() => setView(key)}
                aria-pressed={view === key}
                className={cn("flex items-center gap-1.5 rounded-full px-3 py-1.5 text-[13px] font-medium transition", view === key ? "bg-glacier text-white" : "text-muted hover:text-fg")}
              >
                <Icon className="h-3.5 w-3.5" /> {title}
              </button>
            ))}
          </div>
        }
      />
      {!plan ? (
        <Card><Empty icon={Box} title="No house to draw">The console was started without a floor plan. It comes from <code>house</code> in the configuration.</Empty></Card>
      ) : !webgl ? (
        <Card><Empty icon={Box} title="This browser can't draw 3D">WebGL is switched off or unavailable here. Every device is still on the Remotes page.</Empty></Card>
      ) : (
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_380px]">
          <div className="card relative h-[64vh] min-h-[420px] overflow-hidden rounded-[22px] bg-gradient-to-b from-slate-100 to-slate-300 dark:from-slate-800 dark:to-slate-950">
            <Canvas camera={{ position: [9, 10, 11], fov: 45 }} dpr={[1, 2]}>
              <CameraRig view={view} span={Math.max(frameOf(plan).width, frameOf(plan).depth)} />
              <HouseScene plan={plan} devices={devices} acMode={acMode} roomC={roomC} thermalRoom={thermalRoom} selected={selected} onSelect={setSelected} />
            </Canvas>
            <div className="pointer-events-none absolute bottom-3 left-3 flex flex-wrap gap-2 text-[12px]">
              {roomC !== undefined && <Chip>Room {roomC.toFixed(1)} °C</Chip>}
              {outsideC !== undefined && <Chip>Outside {outsideC.toFixed(1)} °C</Chip>}
              {acMode && <Chip>AC {acMode.toLowerCase()}</Chip>}
              {robot?.state.doing ? <Chip>Robot: {String(robot.state.doing)}</Chip> : null}
            </div>
          </div>
          <div className="grid content-start gap-4">
            {chosen ? (
              <RemoteTile device={chosen} />
            ) : (
              <Card>
                <Empty icon={MousePointerClick} title="Tap a device">
                  Its remote opens here. Drag to turn the house, scroll or pinch to zoom.
                </Empty>
              </Card>
            )}
            {robot && robot.device_id !== chosen?.device_id && <RemoteTile device={robot} />}
          </div>
        </div>
      )}
    </div>
  );
}

function Chip({ children }: { children: React.ReactNode }) {
  return <span className="rounded-full bg-black/55 px-2.5 py-1 font-medium text-white backdrop-blur">{children}</span>;
}

/** Moves the camera to the view chosen, then leaves it to the person's hands. */
function CameraRig({ view, span }: { view: View; span: number }) {
  const controls = useRef<OrbitControlsImpl>(null);
  const { camera } = useThree();
  const moving = useRef(true);
  const last = useRef<View>(view);
  if (last.current !== view) {
    last.current = view;
    moving.current = true;
  }
  const goal = useMemo(
    () => (view === "top" ? new THREE.Vector3(0, span * 1.35, 0.001) : new THREE.Vector3(span * 0.75, span * 0.85, span * 0.95)),
    [view, span],
  );
  useFrame((_, dt) => {
    if (!moving.current) return;
    camera.position.lerp(goal, Math.min(1, dt * 4));
    controls.current?.target.lerp(new THREE.Vector3(0, 0, 0), Math.min(1, dt * 4));
    controls.current?.update();
    if (camera.position.distanceTo(goal) < 0.02) moving.current = false;
  });
  return (
    <OrbitControls
      ref={controls}
      makeDefault
      enableRotate={view === "3d"}
      maxPolarAngle={view === "3d" ? Math.PI / 2.15 : 0.0001}
      minDistance={4}
      maxDistance={span * 2.5}
      onStart={() => { moving.current = false; }}
    />
  );
}

function hasWebGl(): boolean {
  try {
    const canvas = document.createElement("canvas");
    return Boolean(canvas.getContext("webgl2") ?? canvas.getContext("webgl"));
  } catch {
    return false;
  }
}
