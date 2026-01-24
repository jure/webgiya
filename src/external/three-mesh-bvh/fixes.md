Shader validation error: Function [1] 'intersectTriangles' is invalid
    ┌─ compute_Surfel Integrate Pass:216:1
    │  
216 │ ╭ fn intersectTriangles ( bvh_position: ptr<storage, array<vec3f>, read>,
217 │ │         bvh_index: ptr<storage, array<vec3u>, read>,
218 │ │         offset: u32,
219 │ │         count: u32,
    · │
244 │ │ 
245 │ │         return closestResult;
    │ ╰─────────────────────────────^ naga::ir::Function [1]
    │  
    = Argument 'bvh_position' at index 0 is a pointer of space Storage { access: StorageAccess(LOAD) }, which can't be passed into functions.


    ptr is a reserved word