import { Html, Sparkles } from "@react-three/drei";
import { useFrame, type ThreeEvent } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import type { Device, HousePlace, HousePlan, HouseRoom } from "@/lib/types";
import { frameOf, temperatureColour, wallsOf, type Frame, type Vec3 } from "./plan";

// The house, drawn from the board (FR-99). Nothing here decides anything:
// every moving part follows a device's published state -- a fan spins because
// its service says it is on, the robot walks the leg its service says it is
// walking -- so what is seen is what the services are doing.

export interface SceneProps {
  plan: HousePlan;
  devices: Device[];
  /** The thermal room's air conditioner, as the actuator reports it: COOL, OFF... */
  acMode: string | null;
  /** The room's temperature, from the thermal model's sensor. */
  roomC: number | undefined;
  /** The room the thermal model simulates, which the temperature tints. */
  thermalRoom: string;
  selected: string | null;
  onSelect: (deviceId: string | null) => void;
}

export function HouseScene({ plan, devices, acMode, roomC, thermalRoom, selected, onSelect }: SceneProps) {
  const frame = useMemo(() => frameOf(plan), [plan]);
  const byId = useMemo(() => new Map(devices.map((device) => [device.device_id, device])), [devices]);
  const robot = devices.find((device) => device.manifest.kind === "robot");
  return (
    <group onPointerMissed={() => onSelect(null)}>
      <ambientLight intensity={0.9} />
      <hemisphereLight args={["#ffffff", "#b8c4d0", 0.8]} />
      <directionalLight position={[6, 12, 8]} intensity={1.2} />
      {plan.rooms.map((room) => (
        <Floor key={room.room_id} room={room} frame={frame} colour={room.room_id === thermalRoom ? temperatureColour(roomC) : undefined} label={room.room_id === thermalRoom && roomC !== undefined ? `${room.name} · ${roomC.toFixed(1)} °C` : room.name} />
      ))}
      <Walls plan={plan} frame={frame} />
      {plan.rooms.map((room) => <Furniture key={room.room_id} room={room} frame={frame} />)}
      <Spots plan={plan} frame={frame} robot={robot} />
      {Object.entries(plan.placements).map(([deviceId, place]) => {
        const device = byId.get(deviceId);
        const at = frame.at(place.x, place.y, place.z ?? 0);
        const pick = (event: ThreeEvent<MouseEvent>) => {
          event.stopPropagation();
          onSelect(deviceId);
        };
        if (deviceId === "ac" || device?.manifest.kind === "ac") {
          return <Selectable key={deviceId} at={at} onPick={pick} selected={selected === deviceId}><AirConditioner cooling={acMode === "COOL"} heating={acMode === "HEAT"} /></Selectable>;
        }
        if (!device) return null;
        const body = bodyFor(device, place);
        return body && (
          <Selectable key={deviceId} at={at} onPick={pick} selected={selected === deviceId} offline={!device.online} name={device.manifest.name}>
            {body}
          </Selectable>
        );
      })}
      {robot && <Robot device={robot} frame={frame} selected={selected === robot.device_id} onPick={(event) => { event.stopPropagation(); onSelect(robot.device_id); }} />}
    </group>
  );
}

function bodyFor(device: Device, place: HousePlace): React.ReactNode {
  const on = Boolean(device.state.on) && device.online;
  switch (device.manifest.kind) {
    case "fan":
      return <CeilingFan on={on} speed={Number(device.state.speed ?? 0)} />;
    case "light":
      return <Light on={on} brightness={Number(device.state.brightness ?? 100)} height={place.z ?? 0} />;
    case "tv":
      return <Television on={on} channel={Number(device.state.channel ?? 1)} />;
    case "media":
      return <Speaker playing={Boolean(device.state.playing) && device.online} title={String(device.state.track ?? "")} />;
    default:
      return null;
  }
}

// --- the building ----------------------------------------------------------

function Floor({ room, frame, colour, label }: { room: HouseRoom; frame: Frame; colour?: string; label: string }) {
  const [x, , z] = frame.at(room.x + room.width / 2, room.y + room.depth / 2);
  return (
    <group>
      <mesh position={[x, 0, z]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[room.width - 0.02, room.depth - 0.02]} />
        <meshStandardMaterial color={colour ?? FLOOR[room.room_id] ?? "#e7e2da"} />
      </mesh>
      <Html position={[x, 0.02, z - room.depth / 2 + 0.35]} center zIndexRange={[10, 0]} className="pointer-events-none select-none">
        <span className="whitespace-nowrap rounded-full bg-black/45 px-2 py-0.5 text-[11px] font-medium text-white">{label}</span>
      </Html>
    </group>
  );
}

const FLOOR: Record<string, string> = { kitchen: "#e9e1d3", bathroom: "#d9e6ea", bedroom: "#e6dccf", living_room: "#e7e2da" };

function Walls({ plan, frame }: { plan: HousePlan; frame: Frame }) {
  const walls = useMemo(() => wallsOf(plan), [plan]);
  const height = plan.wall_height_m;
  return (
    <group>
      {walls.map((wall, n) => {
        const length = Math.hypot(wall.x1 - wall.x0, wall.y1 - wall.y0);
        const [x, , z] = frame.at((wall.x0 + wall.x1) / 2, (wall.y0 + wall.y1) / 2);
        const along = wall.y0 === wall.y1;
        return (
          <group key={n} position={[x, 0, z]}>
            {/* A solid skirting, and the wall above it as glass, so the rooms can be seen into from any side. */}
            <mesh position={[0, 0.15, 0]}>
              <boxGeometry args={along ? [length, 0.3, 0.1] : [0.1, 0.3, length]} />
              <meshStandardMaterial color="#9aa5b1" />
            </mesh>
            <mesh position={[0, height / 2, 0]}>
              <boxGeometry args={along ? [length, height, 0.06] : [0.06, height, length]} />
              <meshStandardMaterial color="#cfd8e3" transparent opacity={0.16} depthWrite={false} />
            </mesh>
          </group>
        );
      })}
    </group>
  );
}

/** Furniture, placed by fractions of each room so another floor plan still gets some. */
function Furniture({ room, frame }: { room: HouseRoom; frame: Frame }) {
  const at = (fx: number, fy: number, z = 0) => frame.at(room.x + room.width * fx, room.y + room.depth * fy, z);
  const box = (key: string, position: Vec3, size: Vec3, colour: string) => (
    <mesh key={key} position={[position[0], position[1] + size[1] / 2, position[2]]}>
      <boxGeometry args={size} />
      <meshStandardMaterial color={colour} />
    </mesh>
  );
  switch (room.room_id) {
    case "living_room":
      return <group>{[box("sofa", at(0.5, 0.82), [2.2, 0.75, 0.85], "#6b7a8f"), box("table", at(0.5, 0.55), [1.1, 0.4, 0.6], "#a07855"), box("stand", at(0.5, 0.06), [1.6, 0.5, 0.4], "#5a4a3a")]}</group>;
    case "kitchen":
      return <group>{[box("counter", at(0.45, 0.08), [room.width * 0.75, 0.9, 0.6], "#cfcac2"), box("table", at(0.4, 0.55), [1.2, 0.75, 0.8], "#a07855")]}</group>;
    case "bedroom":
      return <group>{[box("bed", at(0.62, 0.55), [2.0, 0.5, 1.6], "#b5a7c9"), box("pillow", at(0.62, 0.3, 0.5), [1.5, 0.12, 0.35], "#f4f1ea"), box("wardrobe", at(0.12, 0.85), [1.2, 1.9, 0.6], "#7a6450")]}</group>;
    case "bathroom":
      return <group>{[box("bath", at(0.3, 0.82), [1.6, 0.55, 0.75], "#f5f7f8"), box("sink", at(0.82, 0.25), [0.6, 0.85, 0.45], "#f5f7f8")]}</group>;
    default:
      return null;
  }
}

// --- devices ---------------------------------------------------------------

function Selectable({ at, onPick, selected, offline, name, children }: { at: Vec3; onPick: (event: ThreeEvent<MouseEvent>) => void; selected: boolean; offline?: boolean; name?: string; children: React.ReactNode }) {
  return (
    <group
      position={at}
      onClick={onPick}
      onPointerOver={(event) => { event.stopPropagation(); document.body.style.cursor = "pointer"; }}
      onPointerOut={() => { document.body.style.cursor = ""; }}
    >
      {children}
      {selected && (
        <mesh position={[0, 0.02 - at[1], 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.45, 0.55, 40]} />
          <meshBasicMaterial color="#3b82f6" />
        </mesh>
      )}
      {offline && (
        <Html position={[0, 0.35, 0]} center zIndexRange={[20, 0]} className="pointer-events-none select-none">
          <span className="whitespace-nowrap rounded-full bg-red-600/85 px-2 py-0.5 text-[10px] font-semibold text-white">{name ?? "device"} offline</span>
        </Html>
      )}
    </group>
  );
}

function CeilingFan({ on, speed }: { on: boolean; speed: number }) {
  const blades = useRef<THREE.Group>(null);
  const rate = useRef(0);
  useFrame((_, dt) => {
    const target = on ? 2 + speed * 5 : 0;
    rate.current += (target - rate.current) * Math.min(1, dt * (on ? 1.2 : 0.6));
    if (blades.current) blades.current.rotation.y += rate.current * dt;
  });
  return (
    <group>
      <mesh position={[0, 0.2, 0]}>
        <cylinderGeometry args={[0.02, 0.02, 0.4, 8]} />
        <meshStandardMaterial color="#6b7280" />
      </mesh>
      <group ref={blades}>
        <mesh>
          <cylinderGeometry args={[0.12, 0.12, 0.1, 20]} />
          <meshStandardMaterial color={on ? "#e5e7eb" : "#9ca3af"} />
        </mesh>
        {[0, 1, 2].map((n) => (
          <mesh key={n} rotation={[0, (n * 2 * Math.PI) / 3, 0]} position={[Math.cos((n * 2 * Math.PI) / 3) * 0.45, 0, -Math.sin((n * 2 * Math.PI) / 3) * 0.45]}>
            <boxGeometry args={[0.7, 0.02, 0.15]} />
            <meshStandardMaterial color="#8b6f4e" />
          </mesh>
        ))}
      </group>
    </group>
  );
}

function Light({ on, brightness, height }: { on: boolean; brightness: number; height: number }) {
  const hanging = height > 1.5;
  const level = on ? Math.max(0.05, brightness / 100) : 0;
  return (
    <group>
      {hanging ? (
        <mesh position={[0, 0.12, 0]}>
          <coneGeometry args={[0.25, 0.18, 24, 1, true]} />
          <meshStandardMaterial color="#f3f4f6" side={THREE.DoubleSide} />
        </mesh>
      ) : (
        <mesh position={[0, -0.35, 0]}>
          <cylinderGeometry args={[0.03, 0.08, 0.7, 12]} />
          <meshStandardMaterial color="#4b5563" />
        </mesh>
      )}
      <mesh>
        <sphereGeometry args={[0.09, 16, 16]} />
        <meshStandardMaterial color={on ? "#fff7d6" : "#9ca3af"} emissive={on ? "#ffd27a" : "#000000"} emissiveIntensity={2.5 * level} />
      </mesh>
      {on && <pointLight intensity={12 * level} distance={6} decay={2} color="#ffdca0" />}
      {on && (
        <mesh position={[0, 0.01 - height, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <circleGeometry args={[1.4 * level + 0.4, 32]} />
          <meshBasicMaterial color="#ffd27a" transparent opacity={0.22 * level} depthWrite={false} />
        </mesh>
      )}
    </group>
  );
}

function Television({ on, channel }: { on: boolean; channel: number }) {
  const screen = useRef<THREE.MeshStandardMaterial>(null);
  useFrame(({ clock }) => {
    if (!screen.current) return;
    if (!on) {
      screen.current.emissive.set("#000000");
      return;
    }
    // A picture that changes, its colour set by the channel.
    const hue = ((channel * 47) % 360) / 360;
    screen.current.emissive.setHSL((hue + Math.sin(clock.elapsedTime * 0.7) * 0.06 + 1) % 1, 0.65, 0.45 + Math.sin(clock.elapsedTime * 2.3) * 0.08);
  });
  return (
    <group>
      <mesh>
        <boxGeometry args={[1.3, 0.76, 0.06]} />
        <meshStandardMaterial color="#111827" />
      </mesh>
      <mesh position={[0, 0, 0.032]}>
        <planeGeometry args={[1.2, 0.66]} />
        <meshStandardMaterial ref={screen} color="#0b0f19" emissive="#000000" emissiveIntensity={1.4} />
      </mesh>
      {on && <pointLight position={[0, 0, 0.6]} intensity={3} distance={3} decay={2} color="#9ec5ff" />}
    </group>
  );
}

const AIR = 60;

function AirConditioner({ cooling, heating }: { cooling: boolean; heating: boolean }) {
  const points = useRef<THREE.Points>(null);
  const seeds = useMemo(() => Array.from({ length: AIR }, () => [Math.random(), Math.random() - 0.5, Math.random()]), []);
  const positions = useMemo(() => new Float32Array(AIR * 3), []);
  const running = cooling || heating;
  useFrame(({ clock }) => {
    if (!points.current) return;
    seeds.forEach(([phase, side, drop], n) => {
      const t = (clock.elapsedTime * 0.5 + phase) % 1;
      positions[n * 3] = 0.2 + t * 2.6;
      positions[n * 3 + 1] = -0.1 - t * t * (0.8 + drop);
      positions[n * 3 + 2] = side * (0.5 + t * 1.4);
    });
    points.current.geometry.attributes.position.needsUpdate = true;
  });
  return (
    <group>
      <mesh>
        <boxGeometry args={[0.24, 0.3, 0.95]} />
        <meshStandardMaterial color="#f8fafc" />
      </mesh>
      <mesh position={[0.125, -0.09, 0]} rotation={[0, 0, running ? -0.5 : 0]}>
        <boxGeometry args={[0.02, 0.06, 0.85]} />
        <meshStandardMaterial color={running ? (cooling ? "#60a5fa" : "#fb923c") : "#cbd5e1"} emissive={running ? (cooling ? "#2563eb" : "#ea580c") : "#000000"} emissiveIntensity={0.6} />
      </mesh>
      <points ref={points} visible={running}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        </bufferGeometry>
        <pointsMaterial size={0.06} color={cooling ? "#93c5fd" : "#fdba74"} transparent opacity={0.85} depthWrite={false} />
      </points>
      <Html position={[0.3, 0.3, 0]} center zIndexRange={[20, 0]} className="pointer-events-none select-none">
        <span className="whitespace-nowrap rounded-full bg-sky-600/85 px-2 py-0.5 text-[10px] font-semibold text-white">AC {cooling ? "cooling" : heating ? "heating" : "off"}</span>
      </Html>
    </group>
  );
}

function Speaker({ playing, title }: { playing: boolean; title: string }) {
  const cone = useRef<THREE.Mesh>(null);
  useFrame(({ clock }) => {
    if (cone.current) cone.current.scale.setScalar(playing ? 1 + Math.abs(Math.sin(clock.elapsedTime * 8)) * 0.18 : 1);
  });
  return (
    <group>
      <mesh position={[0, 0.45, 0]}>
        <boxGeometry args={[0.32, 0.9, 0.3]} />
        <meshStandardMaterial color="#1f2937" />
      </mesh>
      <mesh ref={cone} position={[0, 0.55, 0.16]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.1, 0.06, 0.03, 20]} />
        <meshStandardMaterial color={playing ? "#a78bfa" : "#4b5563"} emissive={playing ? "#7c3aed" : "#000000"} emissiveIntensity={0.8} />
      </mesh>
      {playing && title && (
        <Html position={[0, 1.15, 0]} center zIndexRange={[20, 0]} className="pointer-events-none select-none">
          <span className="whitespace-nowrap rounded-full bg-violet-600/85 px-2 py-0.5 text-[10px] font-semibold text-white">♪ {title}</span>
        </Html>
      )}
    </group>
  );
}

// --- the robot, its dock, the dog --------------------------------------------

/** Where the robot is now, moved along the leg its service last published. */
function useRobotPlace(device: Device) {
  const s = device.state;
  // Only differences between the node's own timestamps are used, so a phone
  // whose clock disagrees with the node's still sees it walk at the right pace.
  const received = useRef({ wall: performance.now() / 1000, elapsed: 0, key: "" });
  const key = `${device.state_ts}:${s.leg_start_ts}`;
  if (received.current.key !== key) {
    received.current = { wall: performance.now() / 1000, elapsed: Number(device.state_ts ?? 0) - Number(s.leg_start_ts ?? 0), key };
  }
  return () => {
    const start = Number(s.leg_start_ts ?? 0);
    const end = Number(s.leg_end_ts ?? 0);
    if (!device.online || end <= start) return { x: Number(s.x ?? 0), y: Number(s.y ?? 0), moving: false };
    const t = Math.max(0, Math.min(1, (received.current.elapsed + performance.now() / 1000 - received.current.wall) / (end - start)));
    const [x0, y0, x1, y1] = [Number(s.leg_from_x), Number(s.leg_from_y), Number(s.leg_to_x), Number(s.leg_to_y)];
    return { x: x0 + (x1 - x0) * t, y: y0 + (y1 - y0) * t, moving: t < 1 };
  };
}

function Robot({ device, frame, selected, onPick }: { device: Device; frame: Frame; selected: boolean; onPick: (event: ThreeEvent<MouseEvent>) => void }) {
  const body = useRef<THREE.Group>(null);
  const brush = useRef<THREE.Mesh>(null);
  const place = useRobotPlace(device);
  const phase = String(device.state.phase ?? "");
  const doing = String(device.state.doing ?? "");
  const progress = Number(device.state.progress ?? 0);
  const battery = Number(device.state.battery ?? 0);
  const heading = (Number(device.state.heading ?? 0) * Math.PI) / 180;
  useFrame(({ clock }, dt) => {
    if (!body.current) return;
    const { x, y, moving } = place();
    const [wx, , wz] = frame.at(x, y);
    body.current.position.x += (wx - body.current.position.x) * Math.min(1, dt * 12);
    body.current.position.z += (wz - body.current.position.z) * Math.min(1, dt * 12);
    body.current.position.y = moving ? Math.abs(Math.sin(clock.elapsedTime * 9)) * 0.03 : 0;
    const target = -heading;
    let turn = target - body.current.rotation.y;
    turn = Math.atan2(Math.sin(turn), Math.cos(turn));
    body.current.rotation.y += turn * Math.min(1, dt * 6);
    if (brush.current) brush.current.rotation.y += (phase === "cleaning" ? 18 : 0) * dt;
  });
  return (
    <group
      ref={body}
      onClick={onPick}
      onPointerOver={(event) => { event.stopPropagation(); document.body.style.cursor = "pointer"; }}
      onPointerOut={() => { document.body.style.cursor = ""; }}
    >
      <mesh position={[0, 0.2, 0]}>
        <cylinderGeometry args={[0.22, 0.25, 0.32, 28]} />
        <meshStandardMaterial color={device.online ? "#f1f5f9" : "#94a3b8"} />
      </mesh>
      <mesh position={[0, 0.42, 0]}>
        <sphereGeometry args={[0.17, 24, 16, 0, Math.PI * 2, 0, Math.PI / 2]} />
        <meshStandardMaterial color="#e2e8f0" />
      </mesh>
      {/* Its face points the way it walks. */}
      <mesh position={[0.17, 0.46, 0]}>
        <boxGeometry args={[0.04, 0.05, 0.16]} />
        <meshStandardMaterial color="#0ea5e9" emissive="#0ea5e9" emissiveIntensity={doing ? 2 : 0.6} />
      </mesh>
      <mesh ref={brush} position={[0, 0.03, 0]}>
        <cylinderGeometry args={[0.2, 0.2, 0.03, 6]} />
        <meshStandardMaterial color={phase === "cleaning" ? "#38bdf8" : "#64748b"} />
      </mesh>
      {phase === "feeding" && (
        <mesh position={[0.3, 0.35, 0]} rotation={[0, 0, -0.8]}>
          <boxGeometry args={[0.25, 0.04, 0.06]} />
          <meshStandardMaterial color="#f59e0b" />
        </mesh>
      )}
      <mesh position={[0, 0.01, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.3, 0.36, 40, 1, 0, (Math.PI * 2 * battery) / 100]} />
        <meshBasicMaterial color={battery < 25 ? "#f97316" : "#22c55e"} />
      </mesh>
      {selected && (
        <mesh position={[0, 0.015, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.42, 0.5, 40]} />
          <meshBasicMaterial color="#3b82f6" />
        </mesh>
      )}
      <Html position={[0, 0.85, 0]} center zIndexRange={[30, 0]} className="pointer-events-none select-none">
        <span className="whitespace-nowrap rounded-full bg-slate-900/80 px-2 py-0.5 text-[11px] font-medium text-white">
          {device.online ? (doing ? `${doing[0].toUpperCase()}${doing.slice(1)} · ${progress}%` : `Robot · ${battery}%`) : "Robot offline"}
        </span>
      </Html>
    </group>
  );
}

/** The dock, the dog's bed and bowl, and the dog, which goes to its bowl when the robot fills it. */
function Spots({ plan, frame, robot }: { plan: HousePlan; frame: Frame; robot?: Device }) {
  const spot = (name: string) => plan.spots[name];
  const dock = spot("dock");
  const bed = spot("dog_bed");
  const bowl = spot("dog_bowl");
  const full = Boolean(robot?.state.bowl_full);
  const cleaned = Number(robot?.state.bathroom_cleaned_ts ?? 0);
  const bathroom = plan.rooms.find((room) => room.room_id === "bathroom");
  const sparkle = useFreshness(cleaned, 8000);
  return (
    <group>
      {dock && (
        <mesh position={frame.at(dock.x, dock.y, 0.01)} rotation={[-Math.PI / 2, 0, 0]}>
          <circleGeometry args={[0.38, 32]} />
          <meshStandardMaterial color="#334155" emissive="#0ea5e9" emissiveIntensity={0.25} />
        </mesh>
      )}
      {bed && (
        <mesh position={frame.at(bed.x, bed.y, 0.06)}>
          <cylinderGeometry args={[0.45, 0.5, 0.12, 28]} />
          <meshStandardMaterial color="#9b5d3a" />
        </mesh>
      )}
      {bowl && (
        <group position={frame.at(bowl.x, bowl.y)}>
          <mesh position={[0, 0.06, 0]}>
            <cylinderGeometry args={[0.16, 0.12, 0.12, 24, 1, true]} />
            <meshStandardMaterial color="#dc2626" side={THREE.DoubleSide} />
          </mesh>
          {full && (
            <mesh position={[0, 0.1, 0]}>
              <sphereGeometry args={[0.14, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2]} />
              <meshStandardMaterial color="#8b5a2b" />
            </mesh>
          )}
        </group>
      )}
      {bed && bowl && <TheDog frame={frame} bed={bed} bowl={bowl} hungry={!full} />}
      {bathroom && sparkle && (
        <Sparkles position={frame.at(bathroom.x + bathroom.width / 2, bathroom.y + bathroom.depth / 2, 0.6)} count={60} scale={[bathroom.width, 1, bathroom.depth]} size={6} speed={0.6} color="#7dd3fc" />
      )}
    </group>
  );
}

/** Whether a timestamp changed to a new value within the last while; never on first sight. */
function useFreshness(value: number, forMs: number): boolean {
  const seen = useRef(value);
  const [fresh, setFresh] = useState(false);
  useEffect(() => {
    if (value && value !== seen.current) {
      setFresh(true);
      const timer = window.setTimeout(() => setFresh(false), forMs);
      seen.current = value;
      return () => window.clearTimeout(timer);
    }
    seen.current = value;
  }, [value, forMs]);
  return fresh;
}

function TheDog({ frame, bed, bowl, hungry }: { frame: Frame; bed: HousePlace; bowl: HousePlace; hungry: boolean }) {
  const dog = useRef<THREE.Group>(null);
  const head = useRef<THREE.Group>(null);
  const tail = useRef<THREE.Mesh>(null);
  const atBowl = frame.at(bowl.x - 0.45, bowl.y);
  const atBed = frame.at(bed.x, bed.y, 0.12);
  useFrame(({ clock }, dt) => {
    if (!dog.current) return;
    const target = hungry ? atBed : atBowl;
    const position = dog.current.position;
    const dx = target[0] - position.x;
    const dz = target[2] - position.z;
    const distance = Math.hypot(dx, dz);
    const step = Math.min(distance, 0.9 * dt);
    if (distance > 0.01) {
      position.x += (dx / distance) * step;
      position.z += (dz / distance) * step;
      dog.current.rotation.y = -Math.atan2(dz, dx);
    } else if (!hungry) {
      dog.current.rotation.y = 0;
    }
    position.y += (target[1] - position.y) * Math.min(1, dt * 4);
    const eating = !hungry && distance < 0.05;
    if (head.current) head.current.rotation.z = eating ? -0.5 - Math.abs(Math.sin(clock.elapsedTime * 6)) * 0.3 : 0;
    if (tail.current) tail.current.rotation.y = Math.sin(clock.elapsedTime * (hungry ? 3 : 12)) * 0.6;
  });
  return (
    <group ref={dog} position={atBed}>
      <mesh position={[0, 0.3, 0]}>
        <boxGeometry args={[0.55, 0.24, 0.24]} />
        <meshStandardMaterial color="#c08a4b" />
      </mesh>
      {[[-0.2, -0.08], [-0.2, 0.08], [0.2, -0.08], [0.2, 0.08]].map(([x, z], n) => (
        <mesh key={n} position={[x, 0.1, z]}>
          <boxGeometry args={[0.07, 0.2, 0.07]} />
          <meshStandardMaterial color="#a8743d" />
        </mesh>
      ))}
      <group ref={head} position={[0.3, 0.4, 0]}>
        <mesh position={[0.08, 0, 0]}>
          <boxGeometry args={[0.2, 0.18, 0.18]} />
          <meshStandardMaterial color="#c08a4b" />
        </mesh>
        <mesh position={[0.2, -0.03, 0]}>
          <boxGeometry args={[0.1, 0.08, 0.1]} />
          <meshStandardMaterial color="#3f2a1a" />
        </mesh>
      </group>
      <mesh ref={tail} position={[-0.32, 0.38, 0]} rotation={[0, 0, 0.6]}>
        <boxGeometry args={[0.18, 0.04, 0.04]} />
        <meshStandardMaterial color="#a8743d" />
      </mesh>
    </group>
  );
}
