import * as THREE from "three";

// Floor textures drawn on a canvas here, so the house needs nothing fetched:
// the console runs on an edge node that may have no internet at all.

const cache = new Map<string, THREE.CanvasTexture>();

function texture(key: string, size: number, draw: (g: CanvasRenderingContext2D, size: number) => void): THREE.CanvasTexture {
  const found = cache.get(key);
  if (found) return found;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const g = canvas.getContext("2d")!;
  draw(g, size);
  const made = new THREE.CanvasTexture(canvas);
  made.wrapS = made.wrapT = THREE.RepeatWrapping;
  made.colorSpace = THREE.SRGBColorSpace;
  made.anisotropy = 8;
  cache.set(key, made);
  return made;
}

/** Wooden planks, one tile of the texture covering about 1.6 m of floor. */
export function wood(base: string, dark: string): THREE.CanvasTexture {
  return texture(`wood:${base}:${dark}`, 512, (g, size) => {
    const rows = 8;
    const h = size / rows;
    let seed = 7;
    const random = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    for (let row = 0; row < rows; row++) {
      let x = -random() * size * 0.5;
      while (x < size) {
        const length = size * (0.45 + random() * 0.4);
        const shade = 0.88 + random() * 0.2;
        g.fillStyle = base;
        g.filter = `brightness(${shade})`;
        g.fillRect(x, row * h, length, h);
        g.filter = "none";
        // The grain: a few faint streaks along the plank.
        g.strokeStyle = dark;
        g.globalAlpha = 0.12;
        for (let n = 0; n < 5; n++) {
          const y = row * h + random() * h;
          g.beginPath();
          g.moveTo(x, y);
          g.bezierCurveTo(x + length * 0.3, y + (random() - 0.5) * 4, x + length * 0.6, y + (random() - 0.5) * 4, x + length, y);
          g.stroke();
        }
        g.globalAlpha = 0.5;
        g.fillStyle = dark;
        g.fillRect(x, row * h, 2, h);
        g.globalAlpha = 1;
        x += length;
      }
      g.fillStyle = dark;
      g.globalAlpha = 0.45;
      g.fillRect(0, row * h, size, 1.5);
      g.globalAlpha = 1;
    }
  });
}

/** Square tiles with grout lines; one tile of the texture covers about 1.2 m. */
export function tiles(base: string, grout: string, count = 4): THREE.CanvasTexture {
  return texture(`tiles:${base}:${grout}:${count}`, 512, (g, size) => {
    const step = size / count;
    let seed = 3;
    const random = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    for (let i = 0; i < count; i++) {
      for (let j = 0; j < count; j++) {
        g.fillStyle = base;
        g.filter = `brightness(${0.95 + random() * 0.08})`;
        g.fillRect(i * step, j * step, step, step);
      }
    }
    g.filter = "none";
    g.fillStyle = grout;
    for (let n = 0; n <= count; n++) {
      g.fillRect(n * step - 2, 0, 4, size);
      g.fillRect(0, n * step - 2, size, 4);
    }
  });
}

/** A woven rug's texture: a border and a soft pattern. */
export function rug(base: string, border: string): THREE.CanvasTexture {
  return texture(`rug:${base}:${border}`, 256, (g, size) => {
    g.fillStyle = border;
    g.fillRect(0, 0, size, size);
    g.fillStyle = base;
    g.fillRect(14, 14, size - 28, size - 28);
    g.strokeStyle = border;
    g.globalAlpha = 0.35;
    g.lineWidth = 3;
    for (let n = 1; n < 6; n++) {
      g.strokeRect(14 + n * 16, 14 + n * 16, size - 28 - n * 32, size - 28 - n * 32);
    }
    g.globalAlpha = 1;
  });
}

/** A texture repeated to cover a floor of this size at this many metres per tile. */
export function covering(made: THREE.CanvasTexture, width: number, depth: number, metres: number): THREE.CanvasTexture {
  const copy = made.clone();
  copy.needsUpdate = true;
  copy.repeat.set(width / metres, depth / metres);
  return copy;
}
