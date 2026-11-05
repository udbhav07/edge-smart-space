import { ContactShadows, Environment, Html, Lightformer, RoundedBox, Sparkles } from "@react-three/drei";
import { useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import type { Device, HousePlace, HousePlan, HouseRoom } from "@/lib/types";
import { frameOf, temperatureColour, wallsOf, type Frame, type Vec3 } from "./plan";
import { covering, rug, tiles, wood } from "./textures";

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
  const span = Math.max(frame.width, frame.depth);
  return (
    <group onPointerMissed={() => onSelect(null)}>
      <Lighting span={span} />
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.02, 0]} receiveShadow>
        <planeGeometry args={[span * 4, span * 4]} />
        <meshStandardMaterial color="#d9dcd3" roughness={1} />
      </mesh>
      {plan.rooms.map((room) => (
        <Floor key={room.room_id} room={room} frame={frame} warmth={room.room_id === thermalRoom ? roomC : undefined} />
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
        const body = bodyFor(device, place, plan.wall_height_m);
        return body && (
          <Selectable key={deviceId} at={at} onPick={pick} selected={selected === deviceId} offline={!device.online} name={device.manifest.name}>
            {body}
          </Selectable>
        );
      })}
      {robot && <Robot device={robot} frame={frame} selected={selected === robot.device_id} onPick={(event) => { event.stopPropagation(); onSelect(robot.device_id); }} />}
      <ContactShadows position={[0, 0.005, 0]} scale={span * 1.4} far={1.6} blur={2.2} opacity={0.35} resolution={512} />
    </group>
  );
}

function bodyFor(device: Device, place: HousePlace, ceiling: number): React.ReactNode {
  const on = Boolean(device.state.on) && device.online;
  const height = place.z ?? 0;
  switch (device.manifest.kind) {
    case "fan":
      return <CeilingFan on={on} speed={Number(device.state.speed ?? 0)} drop={Math.max(0.1, ceiling - height)} />;
    case "light":
      return <Light on={on} brightness={Number(device.state.brightness ?? 100)} height={height} drop={Math.max(0.1, ceiling - height)} />;
    case "tv":
      return <Television on={on} channel={Number(device.state.channel ?? 1)} />;
    case "media":
      return <Speaker playing={Boolean(device.state.playing) && device.online} title={String(device.state.track ?? "")} />;
    default:
      return null;
  }
}

// --- light -------------------------------------------------------------------

function Lighting({ span }: { span: number }) {
  return (
    <>
      <ambientLight intensity={0.25} />
      <hemisphereLight args={["#fffaf0", "#7d756a", 0.55]} />
      <directionalLight
        position={[span * 0.6, span * 1.2, span * 0.45]}
        intensity={2.4}
        color="#fff4e2"
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-bias={-0.0004}
        shadow-normalBias={0.02}
        shadow-camera-left={-span * 0.8}
        shadow-camera-right={span * 0.8}
        shadow-camera-top={span * 0.8}
        shadow-camera-bottom={-span * 0.8}
      />
      {/* Soft reflections from a studio of light panels, built here: nothing is downloaded. */}
      <Environment resolution={128} frames={1}>
        <Lightformer intensity={1.6} position={[0, 6, 0]} rotation-x={Math.PI / 2} scale={[12, 12, 1]} />
        <Lightformer intensity={0.8} position={[-8, 3, 4]} rotation-y={Math.PI / 2} scale={[8, 3, 1]} color="#ffe8cc" />
        <Lightformer intensity={0.6} position={[8, 3, -4]} rotation-y={-Math.PI / 2} scale={[8, 3, 1]} color="#dbeafe" />
      </Environment>
    </>
  );
}

// --- the building ----------------------------------------------------------

const FLOORING: Record<string, () => { map: THREE.CanvasTexture; metres: number; roughness: number }> = {
  living_room: () => ({ map: wood("#b88a5a", "#6b4a2b"), metres: 1.6, roughness: 0.55 }),
  bedroom: () => ({ map: wood("#c9a27a", "#7a5634"), metres: 1.6, roughness: 0.6 }),
  kitchen: () => ({ map: tiles("#e8e2d6", "#b9b1a3", 4), metres: 1.2, roughness: 0.35 }),
  bathroom: () => ({ map: tiles("#dfe9ec", "#9fb3ba", 6), metres: 1.2, roughness: 0.25 }),
};

function Floor({ room, frame, warmth }: { room: HouseRoom; frame: Frame; warmth?: number }) {
  const [x, , z] = frame.at(room.x + room.width / 2, room.y + room.depth / 2);
  const flooring = useMemo(() => {
    const made = (FLOORING[room.room_id] ?? FLOORING.living_room)();
    return { ...made, map: covering(made.map, room.width, room.depth, made.metres) };
  }, [room]);
  return (
    <group>
      <mesh position={[x, 0, z]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[room.width, room.depth]} />
        <meshStandardMaterial map={flooring.map} roughness={flooring.roughness} />
      </mesh>
      {warmth !== undefined && (
        // The room's measured temperature, as a wash of colour over its floor.
        <mesh position={[x, 0.004, z]} rotation={[-Math.PI / 2, 0, 0]}>
          <planeGeometry args={[room.width, room.depth]} />
          <meshBasicMaterial color={temperatureColour(warmth)} transparent opacity={0.22} depthWrite={false} />
        </mesh>
      )}
      <Html position={frame.at(room.x + 0.3, room.y + 0.3, 0.02)} zIndexRange={[10, 0]} className="pointer-events-none select-none">
        <span className="whitespace-nowrap rounded-md bg-white/85 px-1.5 py-0.5 text-[11px] font-semibold text-slate-700 shadow-sm">
          {room.name}{warmth !== undefined ? ` · ${warmth.toFixed(1)} °C` : ""}
        </span>
      </Html>
    </group>
  );
}

const WALL_THICKNESS = 0.14;
/** How tall a wall is drawn while it stands between the viewer and the room. */
const CUT_HEIGHT = 0.32;
/** How tall a wall inside the house is drawn, so every room can be seen into. */
const INSIDE_HEIGHT = 0.85;

function Walls({ plan, frame }: { plan: HousePlan; frame: Frame }) {
  const walls = useMemo(() => wallsOf(plan), [plan]);
  const groups = useRef<(THREE.Group | null)[]>([]);
  const { camera } = useThree();
  const height = plan.wall_height_m;
  const centres = useMemo(() => walls.map((wall) => frame.at((wall.x0 + wall.x1) / 2, (wall.y0 + wall.y1) / 2)), [walls, frame]);
  const outside = useMemo(() => {
    const xs = plan.rooms.flatMap((room) => [room.x, room.x + room.width]);
    const ys = plan.rooms.flatMap((room) => [room.y, room.y + room.depth]);
    const [left, right, top, bottom] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const on = (value: number, edge: number) => Math.abs(value - edge) < 1e-6;
    return walls.map((wall) => (wall.x0 === wall.x1 ? on(wall.x0, left) || on(wall.x0, right) : on(wall.y0, top) || on(wall.y0, bottom)));
  }, [walls, plan]);
  useFrame((_, dt) => {
    // Walls on the viewer's side are cut down, as a dollhouse is, so the
    // rooms can be seen into; from straight above every wall stands.
    const flat = new THREE.Vector2(camera.position.x, camera.position.z);
    const overhead = camera.position.y / Math.max(0.001, camera.position.length()) > 0.97;
    if (flat.lengthSq() > 0) flat.normalize();
    centres.forEach((centre, n) => {
      const group = groups.current[n];
      if (!group) return;
      // Metres towards the viewer: the far outside walls stand, the near ones
      // are cut to a skirting, and the walls inside the house to half height.
      const facing = centre[0] * flat.x + centre[2] * flat.y;
      const goal = overhead || (outside[n] && facing < -1) ? 1 : outside[n] ? CUT_HEIGHT / height : INSIDE_HEIGHT / height;
      group.scale.y += (goal - group.scale.y) * Math.min(1, dt * 6);
    });
  });
  return (
    <group>
      {walls.map((wall, n) => {
        const length = Math.hypot(wall.x1 - wall.x0, wall.y1 - wall.y0) + WALL_THICKNESS;
        const along = wall.y0 === wall.y1;
        return (
          <group key={n} position={[centres[n][0], 0, centres[n][2]]} ref={(group) => { groups.current[n] = group; }}>
            <mesh position={[0, height / 2, 0]} castShadow receiveShadow>
              <boxGeometry args={along ? [length, height, WALL_THICKNESS] : [WALL_THICKNESS, height, length]} />
              <meshStandardMaterial color="#f2eee6" roughness={0.9} />
            </mesh>
            <mesh position={[0, 0.05, 0]}>
              <boxGeometry args={along ? [length + 0.005, 0.1, WALL_THICKNESS + 0.02] : [WALL_THICKNESS + 0.02, 0.1, length + 0.005]} />
              <meshStandardMaterial color="#d8d0c2" roughness={0.7} />
            </mesh>
          </group>
        );
      })}
    </group>
  );
}

// --- furniture -----------------------------------------------------------------

function Solid({ position, size, colour, radius = 0.03, roughness = 0.7, metalness = 0 }: { position: Vec3; size: Vec3; colour: string; radius?: number; roughness?: number; metalness?: number }) {
  return (
    <RoundedBox args={size} radius={Math.min(radius, ...size.map((s) => s / 2 - 0.001))} smoothness={3} position={position} castShadow receiveShadow>
      <meshStandardMaterial color={colour} roughness={roughness} metalness={metalness} />
    </RoundedBox>
  );
}

function Leg({ position, height, colour = "#3f3326" }: { position: Vec3; height: number; colour?: string }) {
  return (
    <mesh position={[position[0], position[1] + height / 2, position[2]]} castShadow>
      <cylinderGeometry args={[0.022, 0.018, height, 10]} />
      <meshStandardMaterial color={colour} roughness={0.5} />
    </mesh>
  );
}

function Rug({ at, size, base, border }: { at: Vec3; size: [number, number]; base: string; border: string }) {
  const map = useMemo(() => rug(base, border), [base, border]);
  return (
    <mesh position={[at[0], 0.006, at[2]]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
      <planeGeometry args={size} />
      <meshStandardMaterial map={map} roughness={1} />
    </mesh>
  );
}

function Plant({ at, scale = 1 }: { at: Vec3; scale?: number }) {
  return (
    <group position={at} scale={scale}>
      <mesh position={[0, 0.18, 0]} castShadow>
        <cylinderGeometry args={[0.16, 0.12, 0.36, 18]} />
        <meshStandardMaterial color="#c2703d" roughness={0.8} />
      </mesh>
      {[[0, 0.62, 0, 0.26], [0.12, 0.5, 0.08, 0.18], [-0.12, 0.52, -0.06, 0.2], [0.04, 0.8, -0.05, 0.17]].map(([x, y, z, r], n) => (
        <mesh key={n} position={[x, y, z]} castShadow>
          <icosahedronGeometry args={[r, 1]} />
          <meshStandardMaterial color={n % 2 ? "#4f7f3a" : "#3f6e30"} roughness={0.85} flatShading />
        </mesh>
      ))}
    </group>
  );
}

/** Furniture, placed by fractions of each room, so another floor plan still gets some. */
function Furniture({ room, frame }: { room: HouseRoom; frame: Frame }) {
  const at = (fx: number, fy: number, z = 0): Vec3 => frame.at(room.x + room.width * fx, room.y + room.depth * fy, z);
  const up = (point: Vec3, y: number): Vec3 => [point[0], y, point[2]];
  switch (room.room_id) {
    case "living_room": {
      const sofa = at(0.5, 0.86);
      const table = at(0.5, 0.6);
      const stand = at(0.5, 0.07);
      return (
        <group>
          <Rug at={at(0.5, 0.62)} size={[3.0, 1.9]} base="#d9cbb3" border="#8c6f52" />
          {/* The sofa: its base, back, arms and three cushions. */}
          <Solid position={up(sofa, 0.22)} size={[2.3, 0.3, 0.85]} colour="#5d6b7a" radius={0.06} roughness={0.95} />
          <Solid position={[sofa[0], 0.55, sofa[2] + 0.32]} size={[2.3, 0.55, 0.22]} colour="#5d6b7a" radius={0.08} roughness={0.95} />
          <Solid position={[sofa[0] - 1.08, 0.45, sofa[2]]} size={[0.18, 0.35, 0.85]} colour="#56636f" radius={0.07} roughness={0.95} />
          <Solid position={[sofa[0] + 1.08, 0.45, sofa[2]]} size={[0.18, 0.35, 0.85]} colour="#56636f" radius={0.07} roughness={0.95} />
          {[-0.65, 0, 0.65].map((dx) => <Solid key={dx} position={[sofa[0] + dx, 0.43, sofa[2] - 0.05]} size={[0.63, 0.14, 0.6]} colour="#6b7a8a" radius={0.06} roughness={1} />)}
          <Solid position={[sofa[0] - 0.7, 0.62, sofa[2] + 0.12]} size={[0.38, 0.3, 0.12]} colour="#d6a15c" radius={0.05} roughness={1} />
          {/* A coffee table on four legs. */}
          <Solid position={up(table, 0.42)} size={[1.15, 0.05, 0.6]} colour="#8a6240" radius={0.015} roughness={0.45} />
          {[[-0.5, -0.24], [0.5, -0.24], [-0.5, 0.24], [0.5, 0.24]].map(([dx, dz]) => <Leg key={`${dx}${dz}`} position={[table[0] + dx, 0, table[2] + dz]} height={0.4} />)}
          {/* The TV's low cabinet. */}
          <Solid position={up(stand, 0.24)} size={[1.8, 0.48, 0.42]} colour="#4a3a2c" radius={0.02} roughness={0.5} />
          <Plant at={at(0.07, 0.1)} />
          <Plant at={at(0.93, 0.12)} scale={0.8} />
        </group>
      );
    }
    case "kitchen": {
      const counter = at(0.42, 0.08);
      const fridge = at(0.9, 0.09);
      const table = at(0.38, 0.62);
      return (
        <group>
          {/* Cabinets along the top wall, a counter top, a hob and a sink. */}
          <Solid position={up(counter, 0.43)} size={[room.width * 0.72, 0.86, 0.6]} colour="#e9e4dc" radius={0.015} roughness={0.6} />
          <Solid position={up(counter, 0.885)} size={[room.width * 0.74, 0.04, 0.64]} colour="#3b3f45" radius={0.01} roughness={0.3} />
          <mesh position={[counter[0] - 0.55, 0.91, counter[2]]} rotation={[-Math.PI / 2, 0, 0]}>
            <planeGeometry args={[0.6, 0.5]} />
            <meshStandardMaterial color="#111111" roughness={0.2} metalness={0.4} />
          </mesh>
          {[[-0.7, -0.1], [-0.4, -0.1], [-0.7, 0.12], [-0.4, 0.12]].map(([dx, dz]) => (
            <mesh key={`${dx}${dz}`} position={[counter[0] + dx, 0.912, counter[2] + dz]} rotation={[-Math.PI / 2, 0, 0]}>
              <ringGeometry args={[0.05, 0.075, 24]} />
              <meshStandardMaterial color="#555555" />
            </mesh>
          ))}
          <Solid position={[counter[0] + 0.55, 0.89, counter[2]]} size={[0.5, 0.03, 0.38]} colour="#9aa4ad" radius={0.01} roughness={0.2} metalness={0.8} />
          <Solid position={up(fridge, 0.95)} size={[0.7, 1.9, 0.68]} colour="#dfe3e6" radius={0.04} roughness={0.25} metalness={0.5} />
          {/* A small table with two chairs. */}
          <mesh position={up(table, 0.74)} castShadow receiveShadow>
            <cylinderGeometry args={[0.5, 0.5, 0.04, 36]} />
            <meshStandardMaterial color="#9c7349" roughness={0.45} />
          </mesh>
          <Leg position={table} height={0.72} />
          {[-0.75, 0.75].map((dx) => (
            <group key={dx}>
              <Solid position={[table[0] + dx, 0.45, table[2]]} size={[0.42, 0.05, 0.42]} colour="#7a5634" radius={0.015} />
              <Solid position={[table[0] + dx * 1.27, 0.7, table[2]]} size={[0.05, 0.5, 0.42]} colour="#7a5634" radius={0.015} />
              {[[-0.17, -0.17], [0.17, -0.17], [-0.17, 0.17], [0.17, 0.17]].map(([lx, lz]) => <Leg key={`${lx}${lz}`} position={[table[0] + dx + lx, 0, table[2] + lz]} height={0.43} />)}
            </group>
          ))}
        </group>
      );
    }
    case "bedroom": {
      const bed = at(0.62, 0.55);
      return (
        <group>
          <Rug at={at(0.62, 0.62)} size={[2.8, 2.4]} base="#e7dfd2" border="#a58f73" />
          {/* The bed: frame, mattress, duvet, pillows and a headboard against the top wall. */}
          <Solid position={up(bed, 0.18)} size={[1.75, 0.3, 2.1]} colour="#6f5440" radius={0.03} roughness={0.6} />
          <Solid position={up(bed, 0.42)} size={[1.65, 0.2, 2.0]} colour="#f4f1ea" radius={0.07} roughness={1} />
          <Solid position={[bed[0], 0.54, bed[2] + 0.25]} size={[1.7, 0.08, 1.45]} colour="#8aa3c2" radius={0.04} roughness={1} />
          {[-0.4, 0.4].map((dx) => <Solid key={dx} position={[bed[0] + dx, 0.58, bed[2] - 0.72]} size={[0.62, 0.14, 0.38]} colour="#ffffff" radius={0.06} roughness={1} />)}
          <Solid position={[bed[0], 0.65, bed[2] - 1.06]} size={[1.85, 1.0, 0.08]} colour="#5b4434" radius={0.03} roughness={0.6} />
          {[-1.25, 1.25].map((dx) => <Solid key={dx} position={[bed[0] + dx, 0.27, bed[2] - 0.8]} size={[0.5, 0.54, 0.42]} colour="#7a5c45" radius={0.02} />)}
          <Solid position={up(at(0.06, 0.3), 0.95)} size={[0.6, 1.9, 1.1]} colour="#e6ddd0" radius={0.02} roughness={0.6} />
          <Plant at={at(0.06, 0.12)} scale={0.9} />
        </group>
      );
    }
    case "bathroom": {
      const bath = at(0.32, 0.84);
      const toilet = at(0.86, 0.75);
      const sink = at(0.78, 0.12);
      return (
        <group>
          {/* A bathtub with water in it. */}
          <Solid position={up(bath, 0.28)} size={[1.65, 0.56, 0.78]} colour="#f7f8f9" radius={0.12} roughness={0.15} />
          <mesh position={[bath[0], 0.53, bath[2]]} rotation={[-Math.PI / 2, 0, 0]}>
            <planeGeometry args={[1.4, 0.56]} />
            <meshStandardMaterial color="#9fd3e6" roughness={0.05} metalness={0.1} transparent opacity={0.85} />
          </mesh>
          {/* A toilet: bowl, seat and tank. */}
          <mesh position={up(toilet, 0.2)} castShadow>
            <cylinderGeometry args={[0.19, 0.15, 0.4, 24]} />
            <meshStandardMaterial color="#f7f8f9" roughness={0.15} />
          </mesh>
          <Solid position={[toilet[0] + 0.2, 0.55, toilet[2]]} size={[0.18, 0.4, 0.42]} colour="#f7f8f9" radius={0.04} roughness={0.15} />
          {/* A vanity with a basin and a mirror. */}
          <Solid position={up(sink, 0.4)} size={[0.8, 0.8, 0.46]} colour="#7c8b90" radius={0.02} roughness={0.5} />
          <Solid position={up(sink, 0.83)} size={[0.84, 0.06, 0.5]} colour="#f7f8f9" radius={0.02} roughness={0.15} />
          <Solid position={[sink[0], 1.45, sink[2] - 0.24]} size={[0.7, 0.8, 0.03]} colour="#dbe9ee" radius={0.01} roughness={0.08} metalness={0.35} />
        </group>
      );
    }
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
        <mesh position={[0, 0.012 - at[1], 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.45, 0.53, 48]} />
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

function CeilingFan({ on, speed, drop }: { on: boolean; speed: number; drop: number }) {
  const blades = useRef<THREE.Group>(null);
  const rate = useRef(0);
  useFrame((_, dt) => {
    const target = on ? 2 + speed * 5 : 0;
    rate.current += (target - rate.current) * Math.min(1, dt * (on ? 1.2 : 0.6));
    if (blades.current) blades.current.rotation.y += rate.current * dt;
  });
  return (
    <group>
      {/* Hung from the ceiling on a rod. */}
      <mesh position={[0, drop / 2, 0]}>
        <cylinderGeometry args={[0.018, 0.018, drop, 10]} />
        <meshStandardMaterial color="#8b8f94" metalness={0.7} roughness={0.3} />
      </mesh>
      <group ref={blades}>
        <mesh castShadow>
          <cylinderGeometry args={[0.13, 0.1, 0.12, 24]} />
          <meshStandardMaterial color="#c9ccd0" metalness={0.6} roughness={0.3} />
        </mesh>
        {[0, 1, 2, 3].map((n) => {
          const angle = (n * Math.PI) / 2;
          return (
            <mesh key={n} rotation={[0.08, -angle, 0]} position={[Math.cos(angle) * 0.47, -0.02, Math.sin(angle) * 0.47]} castShadow>
              <boxGeometry args={[0.72, 0.012, 0.15]} />
              <meshStandardMaterial color="#7a5636" roughness={0.6} />
            </mesh>
          );
        })}
      </group>
    </group>
  );
}

function Light({ on, brightness, height, drop }: { on: boolean; brightness: number; height: number; drop: number }) {
  const hanging = height > 1.5;
  const level = on ? Math.max(0.05, brightness / 100) : 0;
  return (
    <group>
      {hanging ? (
        <>
          <mesh position={[0, drop / 2 + 0.1, 0]}>
            <cylinderGeometry args={[0.006, 0.006, drop, 6]} />
            <meshStandardMaterial color="#333333" />
          </mesh>
          <mesh position={[0, 0.1, 0]} castShadow>
            <coneGeometry args={[0.24, 0.2, 32, 1, true]} />
            <meshStandardMaterial color="#e8e2d6" roughness={0.6} side={THREE.DoubleSide} />
          </mesh>
        </>
      ) : (
        <>
          <mesh position={[0, -height / 2, 0]} castShadow>
            <cylinderGeometry args={[0.015, 0.015, height, 8]} />
            <meshStandardMaterial color="#2f2f2f" metalness={0.6} roughness={0.4} />
          </mesh>
          <mesh position={[0, -height + 0.015, 0]}>
            <cylinderGeometry args={[0.14, 0.14, 0.03, 24]} />
            <meshStandardMaterial color="#2f2f2f" />
          </mesh>
          <mesh position={[0, 0.08, 0]}>
            <cylinderGeometry args={[0.12, 0.17, 0.2, 24, 1, true]} />
            <meshStandardMaterial color="#efe6d4" side={THREE.DoubleSide} emissive={on ? "#ffcf8a" : "#000000"} emissiveIntensity={0.5 * level} />
          </mesh>
        </>
      )}
      <mesh>
        <sphereGeometry args={[0.07, 16, 16]} />
        <meshStandardMaterial color={on ? "#fff7d6" : "#b8b8b8"} emissive={on ? "#ffd27a" : "#000000"} emissiveIntensity={4 * level} toneMapped={false} />
      </mesh>
      {on && <pointLight position={[0, -0.1, 0]} intensity={14 * level} distance={6} decay={2} color="#ffd9a0" />}
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
    screen.current.emissive.setHSL((hue + Math.sin(clock.elapsedTime * 0.7) * 0.06 + 1) % 1, 0.6, 0.42 + Math.sin(clock.elapsedTime * 2.3) * 0.08);
  });
  return (
    <group>
      <Solid position={[0, 0, 0]} size={[1.34, 0.78, 0.05]} colour="#121417" radius={0.015} roughness={0.3} metalness={0.4} />
      <mesh position={[0, 0.01, 0.027]}>
        <planeGeometry args={[1.26, 0.7]} />
        <meshStandardMaterial ref={screen} color="#06080c" emissive="#000000" emissiveIntensity={1.3} roughness={0.5} envMapIntensity={0.15} />
      </mesh>
      {on && <pointLight position={[0, 0, 0.7]} intensity={4} distance={3.5} decay={2} color="#a8c8ff" />}
    </group>
  );
}

const AIR = 70;

function AirConditioner({ cooling, heating }: { cooling: boolean; heating: boolean }) {
  const points = useRef<THREE.Points>(null);
  const vane = useRef<THREE.Mesh>(null);
  const seeds = useMemo(() => Array.from({ length: AIR }, () => [Math.random(), Math.random() - 0.5, Math.random()]), []);
  const positions = useMemo(() => new Float32Array(AIR * 3), []);
  const running = cooling || heating;
  useFrame(({ clock }, dt) => {
    if (vane.current) vane.current.rotation.z += ((running ? -0.6 : 0) - vane.current.rotation.z) * Math.min(1, dt * 3);
    if (!points.current) return;
    seeds.forEach(([phase, side, drop], n) => {
      const t = (clock.elapsedTime * 0.45 + phase) % 1;
      positions[n * 3] = 0.25 + t * 2.8;
      positions[n * 3 + 1] = -0.18 - t * t * (0.9 + drop);
      positions[n * 3 + 2] = side * (0.6 + t * 1.5);
    });
    points.current.geometry.attributes.position.needsUpdate = true;
  });
  return (
    <group>
      {/* A wall-mounted split unit, facing into the room. */}
      <Solid position={[0.02, 0, 0]} size={[0.24, 0.3, 1.0]} colour="#f8f9fa" radius={0.07} roughness={0.35} />
      <mesh ref={vane} position={[0.13, -0.12, 0]}>
        <boxGeometry args={[0.03, 0.05, 0.86]} />
        <meshStandardMaterial color="#e5e7eb" roughness={0.4} />
      </mesh>
      <mesh position={[0.146, 0.06, 0.38]}>
        <sphereGeometry args={[0.012, 10, 10]} />
        <meshStandardMaterial color={running ? (cooling ? "#38bdf8" : "#fb923c") : "#9ca3af"} emissive={running ? (cooling ? "#0ea5e9" : "#f97316") : "#000000"} emissiveIntensity={3} toneMapped={false} />
      </mesh>
      <points ref={points} visible={running}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        </bufferGeometry>
        <pointsMaterial size={0.05} color={cooling ? "#bfe3ff" : "#fed7aa"} transparent opacity={0.7} depthWrite={false} />
      </points>
      <Html position={[0.35, 0.32, 0]} center zIndexRange={[20, 0]} className="pointer-events-none select-none">
        <span className={`whitespace-nowrap rounded-full px-2 py-0.5 text-[10px] font-semibold text-white ${running ? (cooling ? "bg-sky-600/85" : "bg-orange-600/85") : "bg-slate-600/80"}`}>AC {cooling ? "cooling" : heating ? "heating" : "off"}</span>
      </Html>
    </group>
  );
}

function Speaker({ playing, title }: { playing: boolean; title: string }) {
  const cone = useRef<THREE.Mesh>(null);
  useFrame(({ clock }) => {
    if (cone.current) cone.current.scale.setScalar(playing ? 1 + Math.abs(Math.sin(clock.elapsedTime * 8)) * 0.12 : 1);
  });
  return (
    <group>
      <Solid position={[0, 0.5, 0]} size={[0.3, 1.0, 0.3]} colour="#23272e" radius={0.03} roughness={0.5} />
      <mesh ref={cone} position={[0, 0.62, 0.152]}>
        <circleGeometry args={[0.1, 28]} />
        <meshStandardMaterial color={playing ? "#8b5cf6" : "#3a3f47"} emissive={playing ? "#7c3aed" : "#000000"} emissiveIntensity={0.6} />
      </mesh>
      <mesh position={[0, 0.32, 0.152]}>
        <circleGeometry args={[0.06, 24]} />
        <meshStandardMaterial color="#3a3f47" />
      </mesh>
      {playing && title && (
        <Html position={[0, 1.25, 0]} center zIndexRange={[20, 0]} className="pointer-events-none select-none">
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

const ROBOT_WHITE = "#f4f6f8";
const ROBOT_GREY = "#3d434b";

/** A home robot about a metre tall: wheeled base, body, two arms and a screen face. It faces +X. */
function Robot({ device, frame, selected, onPick }: { device: Device; frame: Frame; selected: boolean; onPick: (event: ThreeEvent<MouseEvent>) => void }) {
  const body = useRef<THREE.Group>(null);
  const torso = useRef<THREE.Group>(null);
  const rightArm = useRef<THREE.Group>(null);
  const leftArm = useRef<THREE.Group>(null);
  const wheels = useRef<THREE.Group>(null);
  const eyes = useRef<THREE.Group>(null);
  const placed = useRef(false);
  const place = useRobotPlace(device);
  const phase = String(device.state.phase ?? "");
  const doing = String(device.state.doing ?? "");
  const progress = Number(device.state.progress ?? 0);
  const battery = Number(device.state.battery ?? 0);
  const heading = (Number(device.state.heading ?? 0) * Math.PI) / 180;
  const working = phase === "feeding" || phase === "pressing";
  useFrame(({ clock }, dt) => {
    if (!body.current) return;
    const { x, y, moving } = place();
    const [wx, , wz] = frame.at(x, y);
    if (!placed.current) {
      // Appear where it is, rather than sliding in from the middle of the house.
      body.current.position.set(wx, 0, wz);
      body.current.rotation.y = -heading;
      placed.current = true;
    }
    const k = Math.min(1, dt * 10);
    body.current.position.x += (wx - body.current.position.x) * k;
    body.current.position.z += (wz - body.current.position.z) * k;
    let turn = -heading - body.current.rotation.y;
    turn = Math.atan2(Math.sin(turn), Math.cos(turn));
    body.current.rotation.y += turn * Math.min(1, dt * 5);
    const time = clock.elapsedTime;
    if (torso.current) torso.current.position.y = moving ? Math.sin(time * 10) * 0.008 : Math.sin(time * 1.6) * 0.004;
    if (wheels.current) wheels.current.children.forEach((wheel) => { wheel.rotation.y -= (moving || phase === "cleaning" ? 9 : 0) * dt; });
    const reach = working ? -1.25 + Math.sin(time * 4) * 0.25 : moving ? Math.sin(time * 5) * 0.25 : 0;
    if (rightArm.current) rightArm.current.rotation.z += (reach - rightArm.current.rotation.z) * Math.min(1, dt * 5);
    if (leftArm.current) leftArm.current.rotation.z += ((moving ? -Math.sin(time * 5) * 0.25 : 0) - leftArm.current.rotation.z) * Math.min(1, dt * 5);
    // It blinks now and then.
    if (eyes.current) eyes.current.scale.y = (time % 4.2) < 0.12 ? 0.15 : 1;
  });
  const eye = device.online ? "#38e1ff" : "#475569";
  return (
    <group
      ref={body}
      onClick={onPick}
      onPointerOver={(event) => { event.stopPropagation(); document.body.style.cursor = "pointer"; }}
      onPointerOut={() => { document.body.style.cursor = ""; }}
    >
      {/* The base on two wheels and a castor. */}
      <Solid position={[0, 0.13, 0]} size={[0.52, 0.16, 0.44]} colour={ROBOT_GREY} radius={0.07} roughness={0.5} />
      <group ref={wheels}>
        {[-0.24, 0.24].map((z) => (
          <mesh key={z} position={[0, 0.09, z]} rotation={[Math.PI / 2, 0, 0]} castShadow>
            <cylinderGeometry args={[0.09, 0.09, 0.05, 24]} />
            <meshStandardMaterial color="#1b1e22" roughness={0.8} />
          </mesh>
        ))}
      </group>
      <mesh position={[0, 0.215, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.2, 0.225, 48]} />
        <meshStandardMaterial color={eye} emissive={eye} emissiveIntensity={doing ? 2.5 : 0.8} toneMapped={false} />
      </mesh>
      <group ref={torso}>
        {/* The body, and its arms at the shoulders. */}
        <Solid position={[0, 0.48, 0]} size={[0.4, 0.52, 0.38]} colour={ROBOT_WHITE} radius={0.12} roughness={0.25} />
        <mesh position={[0.2, 0.5, 0]} rotation={[0, 0, -Math.PI / 2]}>
          <circleGeometry args={[0.07, 24]} />
          <meshStandardMaterial color={ROBOT_GREY} />
        </mesh>
        {[[-0.23, leftArm], [0.23, rightArm]].map(([z, ref]) => (
          <group key={String(z)} position={[0, 0.66, z as number]} ref={ref as React.RefObject<THREE.Group>}>
            <mesh position={[0, -0.16, 0]} castShadow>
              <capsuleGeometry args={[0.045, 0.24, 6, 12]} />
              <meshStandardMaterial color={ROBOT_WHITE} roughness={0.3} />
            </mesh>
            <mesh position={[0, -0.34, 0]} castShadow>
              <sphereGeometry args={[0.055, 16, 12]} />
              <meshStandardMaterial color={ROBOT_GREY} roughness={0.4} />
            </mesh>
          </group>
        ))}
        {/* The neck and the head, a screen with two eyes on it. */}
        <mesh position={[0, 0.8, 0]}>
          <cylinderGeometry args={[0.05, 0.06, 0.1, 16]} />
          <meshStandardMaterial color={ROBOT_GREY} />
        </mesh>
        <Solid position={[0, 0.96, 0]} size={[0.3, 0.24, 0.34]} colour={ROBOT_WHITE} radius={0.09} roughness={0.25} />
        <Solid position={[0.13, 0.96, 0]} size={[0.06, 0.17, 0.27]} colour="#0b1015" radius={0.025} roughness={0.1} />
        <group ref={eyes} position={[0.165, 0.97, 0]}>
          {[-0.06, 0.06].map((z) => (
            <mesh key={z} position={[0, 0, z]} rotation={[0, Math.PI / 2, 0]}>
              <circleGeometry args={[0.028, 20]} />
              <meshStandardMaterial color={eye} emissive={eye} emissiveIntensity={3} toneMapped={false} />
            </mesh>
          ))}
        </group>
        {phase === "feeding" && (
          // A scoop of food in its hand.
          <mesh position={[0.32, 0.5, 0.23]}>
            <sphereGeometry args={[0.06, 12, 10]} />
            <meshStandardMaterial color="#8b5a2b" roughness={0.9} />
          </mesh>
        )}
      </group>
      {/* Its battery as a ring on the floor. */}
      <mesh position={[0, 0.008, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.36, 0.41, 48, 1, 0, (Math.PI * 2 * Math.max(1, battery)) / 100]} />
        <meshBasicMaterial color={battery < 25 ? "#f97316" : "#22c55e"} transparent opacity={0.85} />
      </mesh>
      {selected && (
        <mesh position={[0, 0.01, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.46, 0.54, 48]} />
          <meshBasicMaterial color="#3b82f6" />
        </mesh>
      )}
      <Html position={[0, 1.35, 0]} center zIndexRange={[30, 0]} className="pointer-events-none select-none">
        <span className="whitespace-nowrap rounded-full bg-slate-900/80 px-2 py-0.5 text-[11px] font-medium text-white shadow">
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
        <group position={frame.at(dock.x, dock.y)}>
          <mesh position={[0, 0.01, 0]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
            <circleGeometry args={[0.42, 40]} />
            <meshStandardMaterial color="#2b3138" roughness={0.6} />
          </mesh>
          <Solid position={[0, 0.12, 0.38]} size={[0.5, 0.24, 0.08]} colour="#2b3138" radius={0.03} />
        </group>
      )}
      {bed && (
        <group position={frame.at(bed.x, bed.y)}>
          <mesh position={[0, 0.07, 0]} castShadow receiveShadow>
            <torusGeometry args={[0.42, 0.09, 12, 32]} />
            <meshStandardMaterial color="#8a5a3c" roughness={1} />
          </mesh>
          <mesh position={[0, 0.04, 0]} rotation={[-Math.PI / 2, 0, 0]}>
            <circleGeometry args={[0.42, 32]} />
            <meshStandardMaterial color="#d8c3a5" roughness={1} />
          </mesh>
        </group>
      )}
      {bowl && (
        <group position={frame.at(bowl.x, bowl.y)}>
          <mesh position={[0, 0.05, 0]} castShadow>
            <cylinderGeometry args={[0.16, 0.12, 0.1, 28, 1, true]} />
            <meshStandardMaterial color="#b91c1c" side={THREE.DoubleSide} roughness={0.4} />
          </mesh>
          <mesh position={[0, 0.002, 0]} rotation={[-Math.PI / 2, 0, 0]}>
            <circleGeometry args={[0.12, 24]} />
            <meshStandardMaterial color="#7f1d1d" />
          </mesh>
          {full && (
            <mesh position={[0, 0.06, 0]}>
              <sphereGeometry args={[0.13, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2]} />
              <meshStandardMaterial color="#8b5a2b" roughness={1} />
            </mesh>
          )}
        </group>
      )}
      {bed && bowl && <TheDog frame={frame} bed={bed} bowl={bowl} hungry={!full} />}
      {bathroom && sparkle && (
        <Sparkles position={frame.at(bathroom.x + bathroom.width / 2, bathroom.y + bathroom.depth / 2, 0.6)} count={70} scale={[bathroom.width, 1, bathroom.depth]} size={6} speed={0.6} color="#7dd3fc" />
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

const FUR = "#c99a5b";
const FUR_DARK = "#a97a40";

/** A golden dog that lies on its bed, and walks to its bowl to eat when it is filled. It faces +X. */
function TheDog({ frame, bed, bowl, hungry }: { frame: Frame; bed: HousePlace; bowl: HousePlace; hungry: boolean }) {
  const dog = useRef<THREE.Group>(null);
  const head = useRef<THREE.Group>(null);
  const tail = useRef<THREE.Group>(null);
  const legs = useRef<THREE.Group>(null);
  const atBowl = frame.at(bowl.x - 0.5, bowl.y);
  const atBed = frame.at(bed.x, bed.y, 0.04);
  useFrame(({ clock }, dt) => {
    if (!dog.current) return;
    const target = hungry ? atBed : atBowl;
    const position = dog.current.position;
    const dx = target[0] - position.x;
    const dz = target[2] - position.z;
    const distance = Math.hypot(dx, dz);
    const step = Math.min(distance, 0.9 * dt);
    const walking = distance > 0.02;
    if (walking) {
      position.x += (dx / distance) * step;
      position.z += (dz / distance) * step;
      dog.current.rotation.y = -Math.atan2(dz, dx);
    } else if (!hungry) {
      dog.current.rotation.y = 0;
    }
    position.y += (target[1] - position.y) * Math.min(1, dt * 4);
    const time = clock.elapsedTime;
    const eating = !hungry && !walking;
    if (head.current) head.current.rotation.z = eating ? -0.6 - Math.abs(Math.sin(time * 6)) * 0.25 : Math.sin(time * 0.8) * 0.05;
    if (tail.current) tail.current.rotation.y = Math.sin(time * (hungry ? 2.5 : 12)) * 0.7;
    if (legs.current) legs.current.children.forEach((leg, n) => { leg.rotation.z = walking ? Math.sin(time * 12 + (n % 2 ? Math.PI : 0)) * 0.45 : 0; });
  });
  return (
    <group ref={dog} position={atBed}>
      <mesh position={[0, 0.36, 0]} rotation={[0, 0, Math.PI / 2]} castShadow>
        <capsuleGeometry args={[0.12, 0.34, 8, 16]} />
        <meshStandardMaterial color={FUR} roughness={0.9} />
      </mesh>
      <group ref={legs}>
        {[[-0.17, -0.08], [-0.17, 0.08], [0.17, -0.08], [0.17, 0.08]].map(([x, z], n) => (
          <group key={n} position={[x, 0.3, z]}>
            <mesh position={[0, -0.13, 0]} castShadow>
              <capsuleGeometry args={[0.035, 0.2, 4, 10]} />
              <meshStandardMaterial color={FUR_DARK} roughness={0.9} />
            </mesh>
          </group>
        ))}
      </group>
      <group ref={head} position={[0.27, 0.46, 0]}>
        <mesh position={[0.07, 0.04, 0]} castShadow>
          <sphereGeometry args={[0.11, 20, 16]} />
          <meshStandardMaterial color={FUR} roughness={0.9} />
        </mesh>
        <mesh position={[0.17, 0.0, 0]} rotation={[0, 0, Math.PI / 2]} castShadow>
          <capsuleGeometry args={[0.05, 0.06, 6, 12]} />
          <meshStandardMaterial color={FUR} roughness={0.9} />
        </mesh>
        <mesh position={[0.235, 0.01, 0]}>
          <sphereGeometry args={[0.022, 10, 10]} />
          <meshStandardMaterial color="#1a1a1a" roughness={0.3} />
        </mesh>
        {[-0.05, 0.05].map((z) => (
          <mesh key={z} position={[0.15, 0.08, z]}>
            <sphereGeometry args={[0.016, 10, 10]} />
            <meshStandardMaterial color="#1a1a1a" roughness={0.2} />
          </mesh>
        ))}
        {[-0.1, 0.1].map((z) => (
          <mesh key={z} position={[0.04, 0.06, z]} rotation={[z > 0 ? 0.35 : -0.35, 0, 0]} castShadow>
            <boxGeometry args={[0.08, 0.13, 0.025]} />
            <meshStandardMaterial color={FUR_DARK} roughness={0.9} />
          </mesh>
        ))}
      </group>
      <group ref={tail} position={[-0.27, 0.42, 0]}>
        <mesh position={[-0.09, 0.05, 0]} rotation={[0, 0, 0.9]} castShadow>
          <capsuleGeometry args={[0.025, 0.16, 4, 8]} />
          <meshStandardMaterial color={FUR} roughness={0.9} />
        </mesh>
      </group>
    </group>
  );
}
