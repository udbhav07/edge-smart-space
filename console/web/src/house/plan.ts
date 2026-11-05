import type { HousePlan, HouseRoom } from "@/lib/types";

// The floor plan is in metres seen from above: x left to right, y top to
// bottom, as the configuration writes it (FR-99). The scene is three.js's:
// X across, Y up, Z towards the viewer. A plan point (x, y) at height z is
// the scene point (x - cx, z, y - cy), so the house sits centred on the origin.

export type Vec3 = [number, number, number];

export interface Frame {
  cx: number;
  cy: number;
  width: number;
  depth: number;
  at: (x: number, y: number, z?: number) => Vec3;
}

export function frameOf(plan: HousePlan): Frame {
  const xs = plan.rooms.flatMap((room) => [room.x, room.x + room.width]);
  const ys = plan.rooms.flatMap((room) => [room.y, room.y + room.depth]);
  const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  return { cx, cy, width: maxX - minX, depth: maxY - minY, at: (x, y, z = 0) => [x - cx, z, y - cy] };
}

/** How wide a doorway is drawn, in m. */
export const DOOR_WIDTH_M = 0.9;

export interface Wall {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/**
 * Every room's four walls, with a gap wherever a door stands on one. Two rooms
 * that share a wall both draw it; drawn in the same place and colour, the two
 * read as one.
 */
export function wallsOf(plan: HousePlan): Wall[] {
  const walls: Wall[] = [];
  for (const room of plan.rooms) {
    for (const edge of edgesOf(room)) walls.push(...withDoorways(edge, plan));
  }
  return walls;
}

function edgesOf(room: HouseRoom): Wall[] {
  const [left, top, right, bottom] = [room.x, room.y, room.x + room.width, room.y + room.depth];
  return [
    { x0: left, y0: top, x1: right, y1: top },
    { x0: left, y0: bottom, x1: right, y1: bottom },
    { x0: left, y0: top, x1: left, y1: bottom },
    { x0: right, y0: top, x1: right, y1: bottom },
  ];
}

function withDoorways(edge: Wall, plan: HousePlan): Wall[] {
  const horizontal = edge.y0 === edge.y1;
  const along = (point: { x: number; y: number }) => (horizontal ? point.x : point.y);
  const [start, end] = horizontal ? [edge.x0, edge.x1] : [edge.y0, edge.y1];
  const doors = plan.doors
    .filter((door) => (horizontal ? Math.abs(door.y - edge.y0) < 1e-6 : Math.abs(door.x - edge.x0) < 1e-6))
    .map(along)
    .filter((position) => position > start && position < end)
    .sort((a, b) => a - b);
  const pieces: Wall[] = [];
  let from = start;
  for (const door of doors) {
    const to = door - DOOR_WIDTH_M / 2;
    if (to > from) pieces.push(piece(edge, horizontal, from, to));
    from = door + DOOR_WIDTH_M / 2;
  }
  if (end > from) pieces.push(piece(edge, horizontal, from, end));
  return pieces;
}

function piece(edge: Wall, horizontal: boolean, from: number, to: number): Wall {
  return horizontal ? { x0: from, y0: edge.y0, x1: to, y1: edge.y0 } : { x0: edge.x0, y0: from, x1: edge.x0, y1: to };
}

/** A room's floor colour for a temperature: blue when cool, neutral near 25 C, orange when hot. */
export function temperatureColour(celsius: number | undefined): string {
  if (celsius === undefined || !Number.isFinite(celsius)) return "#d8dde3";
  const cool = [96, 165, 250];
  const mild = [216, 221, 227];
  const hot = [251, 146, 60];
  const t = Math.max(-1, Math.min(1, (celsius - 25) / 5));
  const [a, b, share] = t < 0 ? [mild, cool, -t] : [mild, hot, t];
  const mix = a.map((value, n) => Math.round(value + (b[n] - value) * share));
  return `rgb(${mix[0]}, ${mix[1]}, ${mix[2]})`;
}
