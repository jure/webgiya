# Webgiya

https://github.com/user-attachments/assets/d7cb4dbf-b979-4974-a1fe-c8a9f15c90ed

Surfel-based global illumination for the web, built on WebGPU and Three.js. The project explores whether surfels (small surface elements with position, normal, and radius) can look good enough and run fast enough for real-time, dynamic indirect lighting on the open web using a fully compute-driven pipeline.

Read the detailed write-up: [Surfel-based global illumination on the web](https://juretriglav.si/surfel-based-global-illumination-on-the-web)

[Live demo](https://jure.github.io/webgiya/)

## Pipeline passes

Each frame runs a sequence of GPU passes that create, organize, light, and resolve surfels. The list below mirrors the actual code paths, with links to the relevant files in this repo.

| Pass | Purpose | Source |
| --- | --- | --- |
| G-Buffer | Rasterize depth, normals, and albedo for surfel spawning and resolve. | [src/gbuffer.ts](https://github.com/jure/webgiya/blob/main/src/gbuffer.ts) |
| Surfel Prepare | Initialize or clear the pool and reset per-frame counters. | [src/surfelPreparePass.ts](https://github.com/jure/webgiya/blob/main/src/surfelPreparePass.ts) |
| Surfel Find Missing | Screen-space analysis to find under-sampled regions and propose new surfels. | [src/surfelFindMissingPass.ts](https://github.com/jure/webgiya/blob/main/src/surfelFindMissingPass.ts) |
| Surfel Allocate | Allocate new surfels from the pool and seed guiding/irradiance. | [src/surfelAllocatePass.ts](https://github.com/jure/webgiya/blob/main/src/surfelAllocatePass.ts) |
| Surfel Age | Age, recycle, and keep alive surfels based on crowding and usage. | [src/surfelAgePass.ts](https://github.com/jure/webgiya/blob/main/src/surfelAgePass.ts) |
| Hash Grid Build | Build the cascaded spatial hash grid (clear, count, prefix-sum, slot). | [src/surfelHashGrid.ts](https://github.com/jure/webgiya/blob/main/src/surfelHashGrid.ts) |
| Integrate | Per-surfel ray tracing via BVH, guided sampling, MSME temporal update, radial depth atlas. | [src/surfelIntegratePass.ts](https://github.com/jure/webgiya/blob/main/src/surfelIntegratePass.ts) |
| Radial Depth | Moment-based visibility used to reduce light leaks in integrate/resolve. | [src/surfelRadialDepth.ts](https://github.com/jure/webgiya/blob/main/src/surfelRadialDepth.ts) |
| Resolve | Per-pixel gather of nearby surfels with spatial/normal weighting and occlusion. | [src/surfelGIResolvePass.ts](https://github.com/jure/webgiya/blob/main/src/surfelGIResolvePass.ts) |
| Composite | Combine direct and indirect lighting and post-process. | [src/main.ts](https://github.com/jure/webgiya/blob/main/src/main.ts) |

## Key modules

- BVH build and GPU ray queries: [src/sceneBvh.ts](https://github.com/jure/webgiya/blob/main/src/sceneBvh.ts) (backed by the vendored three-mesh-bvh WebGPU backend)
- Surfel pool data layout and ping-pong moment buffers: [src/surfelPool.ts](https://github.com/jure/webgiya/blob/main/src/surfelPool.ts)
- UI, scene presets, and debug modes: [src/ui.ts](https://github.com/jure/webgiya/blob/main/src/ui.ts)

## Running locally

```sh
npm install
npm run dev
```

Other useful commands:

```sh
npm run build
npm run lint
npm run format
```

## Notes

`src/external/three-mesh-bvh` is vendored because it contains fixes for Firefox and Safari that are not upstream yet.
