// src/diffuseArray.ts  (NEW)
import * as THREE from 'three/webgpu';

type MatLike = THREE.Material & { map?: THREE.Texture; color?: THREE.Color };

export type DiffuseArrayResult = {
  diffuseArrayTex: THREE.DataArrayTexture;
  materialIdByUUID: Map<string, number>;
  materialCount: number;
};

// --- helpers: linear <-> sRGB (for baking baseColorFactor correctly) ---
function srgbToLinear01(x: number): number {
  // x in [0,1]
  return x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
}
function linearToSrgb01(x: number): number {
  // x in [0,1]
  return x <= 0.0031308 ? x * 12.92 : 1.055 * Math.pow(x, 1 / 2.4) - 0.055;
}
function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

function getMaterialColorLinear(mat: THREE.Material): THREE.Color {
  const m = mat as MatLike;
  return m.color && (m.color as any).isColor
    ? m.color
    : new THREE.Color(1, 1, 1);
}

function getMaterialMap(mat: THREE.Material): THREE.Texture | null {
  const m = mat as MatLike;
  return m.map ?? null;
}

function drawTextureToRGBA8(
  tex: THREE.Texture,
  size: number,
  canvas: any,
  ctx: any,
): Uint8Array {
  const img: any = (tex as any).image ?? (tex as any).source?.data;
  if (!img) {
    const white = new Uint8Array(size * size * 4);
    white.fill(255);
    return white;
  }

  canvas.width = size;
  canvas.height = size;

  ctx.save();

  // NOTE: glTF textures typically have flipY=false already.
  // If you ever hit a mismatch, you can handle tex.flipY here.
  if ((tex as any).flipY) {
    ctx.translate(0, size);
    ctx.scale(1, -1);
  }

  ctx.clearRect(0, 0, size, size);
  ctx.drawImage(img, 0, 0, size, size);
  ctx.restore();

  const data = ctx.getImageData(0, 0, size, size).data; // Uint8ClampedArray
  return new Uint8Array(data);
}

export function buildDiffuseArrayTexture(
  scene: THREE.Scene,
  layerSize = 1024,
): DiffuseArrayResult {
  // 1) Collect unique materials
  const materialIdByUUID = new Map<string, number>();
  const materials: THREE.Material[] = [];

  scene.traverse((obj) => {
    if (!(obj instanceof THREE.Mesh) || !obj.visible) return;

    const mesh = obj as THREE.Mesh;
    const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];

    for (const m of mats) {
      if (!m) continue;
      if (!materialIdByUUID.has(m.uuid)) {
        materialIdByUUID.set(m.uuid, materials.length);
        materials.push(m);
      }
    }
  });

  const materialCount = Math.max(1, materials.length);
  const layerBytes = layerSize * layerSize * 4;
  const all = new Uint8Array(layerBytes * materialCount);

  // reuse one canvas (avoid allocations)
  const canvas: any =
    typeof OffscreenCanvas !== 'undefined'
      ? new OffscreenCanvas(layerSize, layerSize)
      : Object.assign(document.createElement('canvas'), {
          width: layerSize,
          height: layerSize,
        });

  const ctx: any = canvas.getContext('2d', { willReadFrequently: true });

  for (let layer = 0; layer < materialCount; layer++) {
    const mat = materials[layer];
    const baseColorLin = getMaterialColorLinear(mat);
    const map = getMaterialMap(mat);

    let rgba = map
      ? drawTextureToRGBA8(map, layerSize, canvas, ctx)
      : (() => {
          // solid-color layer (encode linear baseColor into sRGB bytes)
          const out = new Uint8Array(layerBytes);
          const r = Math.round(clamp01(linearToSrgb01(baseColorLin.r)) * 255);
          const g = Math.round(clamp01(linearToSrgb01(baseColorLin.g)) * 255);
          const b = Math.round(clamp01(linearToSrgb01(baseColorLin.b)) * 255);
          for (let i = 0; i < layerSize * layerSize; i++) {
            const o = i * 4;
            out[o + 0] = r;
            out[o + 1] = g;
            out[o + 2] = b;
            out[o + 3] = 255;
          }
          return out;
        })();

    // OPTIONAL BUT RECOMMENDED:
    // bake baseColorFactor into the texture layer so we don't need a material table.
    const bake =
      Math.abs(baseColorLin.r - 1) > 1e-4 ||
      Math.abs(baseColorLin.g - 1) > 1e-4 ||
      Math.abs(baseColorLin.b - 1) > 1e-4;

    if (map && bake) {
      // Convert each pixel sRGB->linear, multiply by baseColorLin, linear->sRGB
      for (let i = 0; i < rgba.length; i += 4) {
        const rS = rgba[i + 0] / 255;
        const gS = rgba[i + 1] / 255;
        const bS = rgba[i + 2] / 255;

        const rL = srgbToLinear01(rS) * baseColorLin.r;
        const gL = srgbToLinear01(gS) * baseColorLin.g;
        const bL = srgbToLinear01(bS) * baseColorLin.b;

        rgba[i + 0] = Math.round(clamp01(linearToSrgb01(rL)) * 255);
        rgba[i + 1] = Math.round(clamp01(linearToSrgb01(gL)) * 255);
        rgba[i + 2] = Math.round(clamp01(linearToSrgb01(bL)) * 255);
        // alpha unchanged
      }
    }

    all.set(rgba, layer * layerBytes);
  }

  // 2) Create the array texture
  const tex = new THREE.DataArrayTexture(
    all,
    layerSize,
    layerSize,
    materialCount,
  );
  tex.format = THREE.RGBAFormat;
  tex.type = THREE.UnsignedByteType;

  // IMPORTANT: make this behave like a color texture
  tex.colorSpace = THREE.SRGBColorSpace;

  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.generateMipmaps = false; // keep simple initially
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  tex.name = 'DiffuseArrayTex';

  return { diffuseArrayTex: tex, materialIdByUUID, materialCount };
}
