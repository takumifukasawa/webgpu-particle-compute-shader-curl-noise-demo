// --------------------------------------------------------------------
// ref:
// webgpu
// - https://zenn.dev/emadurandal/books/cb6818fd3a1b2e
// - https://codelabs.developers.google.com/your-first-webgpu-app?hl=ja#0
// - https://webgpufundamentals.org/
// - https://qiita.com/metaphysical_bard/items/db74484d631038bb7ae1
// - https://inzkyk.xyz/misc/webgpu/
// - https://webgpureport.org/
// curl noise
// - https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph2007-curlnoise.pdf
// - https://github.com/IndieVisualLab/UnityGraphicsProgramming2/tree/master
// - https://edom18.hateblo.jp/entry/2018/01/18/081750
// simplex noise
// - https://github.com/stegu/webgl-noise/blob/master/src/noise3D.glsl
// - https://eliotbo.github.io/glsl2wgsl/
// math
// - https://hooktail.sub.jp/vectoranalysis/VectorRotation/
// - https://manabitimes.jp/physics/1962
// --------------------------------------------------------------------

import {
    vec3,
    mat4,
} from 'https://wgpu-matrix.org/dist/3.x/wgpu-matrix.module.js';
import {Pane} from 'https://cdn.jsdelivr.net/npm/tweakpane@4.0.5/dist/tweakpane.min.js';


const main = async () => {
    let width, height;
    let currentTime = -1;
    let deltaTime = 0;
    let needsUpdateDirtyFlag = true;
    let needsResetDirtyFlag = true;
    let beginResetLastTime = -1;
    let requestResetLastTime = -1;
    let cameraPosition = vec3.create(0, 0, 0);

    const instanceVertexElementsNum = 4 + 4;

    const wrapperElement = document.getElementById('js-wrapper');

    const canvas = document.getElementById('js-canvas');
    const context = canvas.getContext('webgpu');

    const gAdapter = await navigator.gpu.requestAdapter(); // 物理デバイス
    const gDevice = await gAdapter.requestDevice(); // 論理デバイス

    // デバイスの制限値を取得
    const limits = gDevice.limits;

    console.log("Max Compute Workgroup Size:", {
        x: limits.maxComputeWorkgroupSizeX,
        y: limits.maxComputeWorkgroupSizeY,
        z: limits.maxComputeWorkgroupSizeZ,
    });

    console.log("Max Compute Workgroups Per Dimension:",
        limits.maxComputeWorkgroupsPerDimension,
    );

    // スレッドを一次元にする
    const workgroupSize = limits.maxComputeWorkgroupSizeX;

    const maxInstanceNum =
        Math.min(
            limits.maxComputeWorkgroupSizeX * limits.maxComputeWorkgroupsPerDimension, // 一次元で実行できる最大数
            Math.pow(2, 32), // js の配列の最大長
        ) / instanceVertexElementsNum; // 1インスタンスあたりの要素数

    console.log("max instance num", maxInstanceNum);

    const parameters = {
        instanceNum: 200000,
        color: '#4f6ff5e5',
        speed: 1,
        noiseScale: 0.05,
        particleScale: 0.1,
        distanceFadePower: 0.025,
    };

    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

    context.configure({
        device: gDevice,
        format: presentationFormat,
        alphaMode: 'opaque',
    });

    // -------------------------------------
    // 0 ------ 3
    // | \      |
    // |  \     |
    // |   \    |
    // |    \   |
    // |     \  |
    // |      \ |
    // 1 ------ 2
    // -------------------------------------

    const quadVertexSize = 4 * 4 + 4 * 4 + 4 * 2; // 頂点ごとのバイトデータ(position, color, uv)
    const quadPositionOffset = 4 * 0; // 座標のオフセット. 先頭なので0. 合計32bit
    const quadColorOffset = 4 * 4; // 頂点カラーのオフセット. 32bitオフセット. 合計32bit
    const quadUVOffset = 4 * 4 + 4 * 4; // UVのオフセット(float4, float4 のオフセット)

    const quadVertexArray = new Float32Array([
        // position(vec4), color(vec4), uv(vec2)
        // left top
        -1, 1, 0, 1,
        1, 1, 1, 1,
        0, 1,
        // left bottom
        -1, -1, 0, 1,
        1, 1, 1, 1,
        0, 0,
        // right bottom
        1, -1, 0, 1,
        1, 1, 1, 1,
        1, 0,
        // right top
        1, 1, 0, 1,
        1, 1, 1, 1,
        1, 1,
    ]);

    const particleVerticesBuffer = gDevice.createBuffer({
        size: quadVertexArray.byteLength,
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true,
    });

    new Float32Array(particleVerticesBuffer.getMappedRange()).set(quadVertexArray);
    particleVerticesBuffer.unmap();

    // position(vec4), velocity(vec4)
    let instanceDataArray;

    const initializeInstances = () => {
        instanceDataArray = new Float32Array(
            new Array(maxInstanceNum)
                .fill(0)
                .map(() => {
                    const s = 5;
                    const v = 128;
                    return [
                        // position
                        Math.random() * s - s / 2,
                        Math.random() * s - s / 2,
                        Math.random() * s - s / 2,
                        1,
                        // velocity
                        Math.random() * v - v / 2,
                        Math.random() * v - v / 2,
                        Math.random() * v - v / 2,
                        1,
                    ];
                })
                .flat()
        );
    }

    initializeInstances();

    // 反時計回りでインデックスを貼る
    const quadIndexArray = new Uint16Array([0, 1, 2, 0, 2, 3]);
    const quadIndicesBuffer = gDevice.createBuffer({
        size: quadIndexArray.byteLength,
        usage: GPUBufferUsage.INDEX,
        mappedAtCreation: true,
    });

    new Uint16Array(quadIndicesBuffer.getMappedRange()).set(quadIndexArray);
    quadIndicesBuffer.unmap();

    const vertWGSL = `
    struct Uniforms {
        projectionMatrix : mat4x4<f32>,
        viewMatrix : mat4x4<f32>,
        worldMatrix : mat4x4<f32>,
        color: vec4<f32>,
        particleMisc: vec2<f32>, // [particleScale, distanceFadePower]
    };

    struct Particle {
        position: vec4<f32>,
        velocity: vec4<f32>
    };
    
    @group(0) @binding(0) 
    var<uniform> uniforms : Uniforms;
    
    @group(1) @binding(0) 
    var<storage, read> instanceData : array<Particle>;
        
    struct VertexOutput {
        @builtin(position) Position : vec4<f32>, // positionを出力するのは必須
        @location(0) fragUV : vec2<f32>,
        @location(1) fragColor: vec4<f32>,
        @location(2) distanceFadeRate : f32
    }
    
    @vertex 
    fn main(
        @location(0) position : vec4<f32>, // ビルボードのオフセット方向
        @location(1) color : vec4<f32>,
        @location(2) uv: vec2<f32>,
        @location(3) instancePosition: vec4<f32>,
        @location(4) instanceVelocity: vec4<f32>
    ) -> VertexOutput {
        var output : VertexOutput;
        
        let particleScale = uniforms.particleMisc.x;
        let distanceFadePower = uniforms.particleMisc.y;
        
        let viewPosition =
            uniforms.viewMatrix *
            uniforms.worldMatrix *
            vec4<f32>(instancePosition.xyz, 1.);

        // view座標系ベースでビルボードの大きさ計算
        output.Position =
            uniforms.projectionMatrix *
            vec4<f32>(
                viewPosition.xy + (position.xy * particleScale),
                viewPosition.zw
            );
        output.fragColor = color * uniforms.color;
        output.fragUV = uv;

        let cameraDistance = length(viewPosition.xyz);
        output.distanceFadeRate = 1 / exp(cameraDistance * distanceFadePower);

        return output;
    }
    `;

    const fragWGSL = `
    @fragment
    fn main(
        @location(0) fragUV: vec2<f32>,
        @location(1) fragColor: vec4<f32>,
        @location(2) distanceFadeRate: f32
    ) -> @location(0) vec4<f32> {
        let len = length(fragUV - vec2<f32>(0.5));
        let alpha = (1. - smoothstep(.25, .5, len)) * .25 * distanceFadeRate;
        let color = vec4<f32>(fragColor.rgb, alpha * fragColor.a);
        return color;
    }
    `;

    const particleRenderPipeline = gDevice.createRenderPipeline({
        layout: 'auto',
        vertex: {
            module: gDevice.createShaderModule({
                code: vertWGSL,
            }),
            entryPoint: 'main',
            buffers: [
                {
                    arrayStride: quadVertexSize,
                    stepMode: 'vertex',
                    attributes: [
                        {
                            shaderLocation: 0, // @location(0)
                            offset: quadPositionOffset,
                            format: 'float32x4'
                        },
                        {
                            shaderLocation: 1, // @location(1)
                            offset: quadColorOffset,
                            format: 'float32x4'
                        },
                        {
                            shaderLocation: 2, // @location(2)
                            offset: quadUVOffset,
                            format: 'float32x2'
                        }
                    ]
                },
                {
                    arrayStride: 4 * 4 + 4 * 4,
                    stepMode: 'instance',
                    attributes: [
                        {
                            shaderLocation: 3, // @location(3)
                            offset: 0,
                            format: 'float32x4'
                        },
                        {
                            shaderLocation: 4, // @location(4)
                            offset: 4 * 4,
                            format: 'float32x4'
                        }
                    ]
                }
            ]
        },
        fragment: {
            module: gDevice.createShaderModule({
                code: fragWGSL,
            }),
            entryPoint: 'main',
            targets: [
                // @location(0)
                {
                    format: presentationFormat,
                    blend: {
                        color: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one',
                            operation: 'add',
                        },
                        alpha: {
                            srcFactor: 'zero',
                            dstFactor: 'one',
                            operation: 'add',
                        }
                    }
                }
            ]
        },
        primitive: {
            topology: 'triangle-list',
        },
    });

    const particleInstancesBuffer = gDevice.createBuffer({
        size: instanceDataArray.byteLength,
        usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.VERTEX |
            GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    new Float32Array(particleInstancesBuffer.getMappedRange()).set(instanceDataArray);
    particleInstancesBuffer.unmap();

    // projectionMatrix : mat4x4<f32>,
    // viewMatrix : mat4x4<f32>,
    // worldMatrix : mat4x4<f32>,
    // color: vec4<f32>
    // misc[particleScale, distanceFadePower, -, -]: vec4<f32>
    const particleUniformBufferSize = 4 * 16 * 3 + 4 * 4 + 4 * 4;
    const particleUniformBuffer = gDevice.createBuffer({
        size: particleUniformBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    const particleUniformBindGroup = gDevice.createBindGroup({
        layout: particleRenderPipeline.getBindGroupLayout(0), // @group(0)
        entries: [
            {
                binding: 0, // @binding(0)
                resource: {
                    buffer: particleUniformBuffer
                }
            },
        ]
    });

    const worldMatrix = mat4.identity();

    gDevice.queue.writeBuffer(
        particleUniformBuffer,
        4 * 16 * 2,
        worldMatrix.buffer,
        worldMatrix.byteOffset,
        worldMatrix.byteLength
    );

    //
    // compute
    //

    gDevice.queue.writeBuffer(particleInstancesBuffer, 0, instanceDataArray);

    const computeShaderModule = gDevice.createShaderModule({
        code: `
            struct Uniforms {
                time: f32,
                deltaTime: f32,
                speed: f32,
                noiseScale: f32,
            };
            
            struct Instance {
                position: vec4<f32>,
                velocity: vec4<f32>
            };

            @group(0) @binding(0)
            var<uniform> uniforms : Uniforms;
            
            @group(0) @binding(1)
            var<storage, read_write> input : array<Instance>;
            
            fn mod289_v3(x: vec3<f32>) -> vec3<f32> {
            	return x - floor(x * (1. / 289.)) * 289.;
            } 
            
            fn mod289_v4(x: vec4<f32>) -> vec4<f32> {
            	return x - floor(x * (1. / 289.)) * 289.;
            } 
    
            fn mod_v3(x: vec3<f32>, y: vec3<f32>) -> vec3<f32> {
                return x - y * floor(x / y);
            }

            fn permute(x: vec4<f32>) -> vec4<f32> {
            	return mod289_v4((x * 34. + 10.) * x);
            } 
            
            fn taylorInvSqrt(r: vec4<f32>) -> vec4<f32> {
            	return 1.7928429 - 0.85373473 * r;
            } 
           
            fn snoise(v: vec3<f32>) -> f32 {
            	let C: vec2<f32> = vec2<f32>(1. / 6., 1. / 3.);
            	let D: vec4<f32> = vec4<f32>(0., 0.5, 1., 2.);
            	var i: vec3<f32> = floor(v + dot(v, C.yyy));
            	let x0: vec3<f32> = v - i + dot(i, C.xxx);
            	let g: vec3<f32> = step(x0.yzx, x0.xyz);
            	let l: vec3<f32> = 1. - g;
            	let i1: vec3<f32> = min(g.xyz, l.zxy);
            	let i2: vec3<f32> = max(g.xyz, l.zxy);
            	let x1: vec3<f32> = x0 - i1 + C.xxx;
            	let x2: vec3<f32> = x0 - i2 + C.yyy;
            	let x3: vec3<f32> = x0 - D.yyy;
            	i = mod289_v3(i);
            	let p: vec4<f32> = permute(permute(permute(i.z + vec4<f32>(0., i1.z, i2.z, 1.)) + i.y + vec4<f32>(0., i1.y, i2.y, 1.)) + i.x + vec4<f32>(0., i1.x, i2.x, 1.));
            	let n_: f32 = 0.14285715;
            	let ns: vec3<f32> = n_ * D.wyz - D.xzx;
            	let j: vec4<f32> = p - 49. * floor(p * ns.z * ns.z);
            	let x_: vec4<f32> = floor(j * ns.z);
            	let y_: vec4<f32> = floor(j - 7. * x_);
            	let x: vec4<f32> = x_ * ns.x + ns.yyyy;
            	let y: vec4<f32> = y_ * ns.x + ns.yyyy;
            	let h: vec4<f32> = 1. - abs(x) - abs(y);
            	let b0: vec4<f32> = vec4<f32>(x.xy, y.xy);
            	let b1: vec4<f32> = vec4<f32>(x.zw, y.zw);
            	let s0: vec4<f32> = floor(b0) * 2. + 1.;
            	let s1: vec4<f32> = floor(b1) * 2. + 1.;
            	let sh: vec4<f32> = -step(h, vec4<f32>(0.));
            	let a0: vec4<f32> = b0.xzyw + s0.xzyw * sh.xxyy;
            	let a1: vec4<f32> = b1.xzyw + s1.xzyw * sh.zzww;
            	var p0: vec3<f32> = vec3<f32>(a0.xy, h.x);
            	var p1: vec3<f32> = vec3<f32>(a0.zw, h.y);
            	var p2: vec3<f32> = vec3<f32>(a1.xy, h.z);
            	var p3: vec3<f32> = vec3<f32>(a1.zw, h.w);
            	let norm: vec4<f32> = taylorInvSqrt(vec4<f32>(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
            	p0 = p0 * (norm.x);
            	p1 = p1 * (norm.y);
            	p2 = p2 * (norm.z);
            	p3 = p3 * (norm.w);
            	var m: vec4<f32> = max(0.5 - vec4<f32>(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), vec4<f32>(0, 0, 0, 0));
            	m = m * m;
            	return 105. * dot(m * m, vec4<f32>(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
            } 
            
            fn snoise3(v: vec3<f32>) -> vec3<f32> {
                return vec3<f32>(
                    snoise(v),
                    snoise(v + vec3<f32>(100., 200., 300.)),
                    snoise(v + vec3<f32>(400., 500., 600.))
                );
            }
            
            fn curlNoise(position: vec3<f32>) -> vec3<f32> {
                let eps = 0.001;
                let eps2 = 2.0 * eps;
                let invEps2 = 1.0 / eps2;
                let dx = vec3<f32>(eps, 0.0, 0.0);
                let dy = vec3<f32>(0.0, eps, 0.0);
                let dz = vec3<f32>(0.0, 0.0, eps);

                let p_x0 = snoise3(position - dx);
                let p_x1 = snoise3(position + dx);
                let p_y0 = snoise3(position - dy);
                let p_y1 = snoise3(position + dy);
                let p_z0 = snoise3(position - dz);
                let p_z1 = snoise3(position + dz);

                let x = (p_y1.z - p_y0.z) - (p_z1.y - p_z0.y);
                let y = (p_z1.x - p_z0.x) - (p_x1.z - p_x0.z);
                let z = (p_x1.y - p_x0.y) - (p_y1.x - p_y0.x);

                return vec3<f32>(x, y, z) * invEps2;
            }

            @compute @workgroup_size(${workgroupSize})
            fn main(
                @builtin(global_invocation_id) global_invocation_id: vec3<u32>
            ) {
                let id = global_invocation_id.x;

                if(id < arrayLength(&input)) {
                    let fid: f32 = f32(id);
                    let instance = input[id];
                    var currentPosition = instance.position.xyz;
                    var currentVelocity = instance.velocity.xyz;
                    let force = curlNoise(currentPosition.xyz * uniforms.noiseScale) - currentVelocity;
                    let newVelocity = force * uniforms.speed * uniforms.deltaTime;
                    let newPosition = currentPosition + newVelocity;
                    input[id].position = vec4<f32>(newPosition.xyz, 1.);
                    input[id].velocity = vec4<f32>(newVelocity.xyz, 1.);
                }
            }
            `,
    });

    const computeBindGroupLayout = gDevice.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: 'uniform',
                }
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: 'storage',
                }
            },

        ]
    });

    const computePipeline = gDevice.createComputePipeline({
        layout: gDevice.createPipelineLayout({
            bindGroupLayouts: [computeBindGroupLayout]
        }),
        compute: {
            module: computeShaderModule,
            entryPoint: 'main'
        }
    });

    // time: f32
    // delta time: f32
    // speed: f32
    // noise scale: f32
    const computeUniformElementsSize = 4;
    const computeUniformBufferSize = 4 * 4;
    const computeUniformBuffer = gDevice.createBuffer({
        size: computeUniformBufferSize,
        usage:
            GPUBufferUsage.UNIFORM |
            GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    new Float32Array(computeUniformBuffer.getMappedRange()).set(new Float32Array(computeUniformElementsSize));
    computeUniformBuffer.unmap();

    const computeBindGroup = gDevice.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0), // @group(0)
        entries: [
            {
                binding: 0, // @binding(0)
                resource: {
                    buffer: computeUniformBuffer
                },
            },
            {
                binding: 1, // @binding(1)
                resource: {
                    buffer: particleInstancesBuffer
                },
            },
        ]
    });

    const update = async () => {
        needsUpdateDirtyFlag = false;

        if (needsResetDirtyFlag) {
            beginResetLastTime = performance.now();
            gDevice.queue.writeBuffer(particleInstancesBuffer, 0, instanceDataArray);
        }

        //
        // compute
        //

        const computeUniformData = new Float32Array([
            currentTime,
            deltaTime,
            parameters.speed,
            parameters.noiseScale,
        ]);
        gDevice.queue.writeBuffer(
            computeUniformBuffer,
            0,
            computeUniformData.buffer,
            computeUniformData.byteOffset,
            computeUniformData.byteLength
        );

        // 一次元で実行
        const dispatchGroupSize =
            Math.min(
                limits.maxComputeWorkgroupsPerDimension,
                Math.ceil(parameters.instanceNum / workgroupSize)
            );

        const computeCommandEncoder = gDevice.createCommandEncoder();
        const computePassEncoder = computeCommandEncoder.beginComputePass();
        computePassEncoder.setPipeline(computePipeline);
        computePassEncoder.setBindGroup(0, computeBindGroup);
        computePassEncoder.dispatchWorkgroups(dispatchGroupSize);
        computePassEncoder.end();
        gDevice.queue.submit([computeCommandEncoder.finish()]);

        //
        // draw
        //

        const colorCoord = parameters.color;
        const hex = colorCoord.substring(1);
        const r = parseInt(hex.substring(0, 2), 16);
        const g = parseInt(hex.substring(2, 4), 16);
        const b = parseInt(hex.substring(4, 6), 16);
        const a = parseInt(hex.substring(6, 8), 16);

        const colorData = new Float32Array([
            r / 255,
            g / 255,
            b / 255,
            a / 255
        ]);
        gDevice.queue.writeBuffer(
            particleUniformBuffer,
            4 * 16 * 3,
            colorData.buffer,
            colorData.byteOffset,
            colorData.byteLength
        );

        const particleMiscData = new Float32Array([
            parameters.particleScale,
            parameters.distanceFadePower,
            0,
            0
        ]);
        gDevice.queue.writeBuffer(
            particleUniformBuffer,
            4 * 16 * 3 + 4 * 4,
            particleMiscData.buffer,
            particleMiscData.byteOffset,
            particleMiscData.byteLength
        );

        // gpuへの命令をパッキングするバッファ
        const renderCommandEncoder = gDevice.createCommandEncoder();

        const renderPassDescriptor = {
            colorAttachments: [
                {
                    view: context.getCurrentTexture().createView(),
                    clearValue: {r: 0.0, g: 0.0, b: 0.0, a: 1.0},
                    loadOp: 'clear', // 命令実施前に行う処理
                    storeOp: 'store', // 命令実施後のバッファの扱い
                }
            ],
        };

        const renderPassEncoder = renderCommandEncoder.beginRenderPass(renderPassDescriptor);
        renderPassEncoder.setPipeline(particleRenderPipeline);
        renderPassEncoder.setBindGroup(0, particleUniformBindGroup);
        renderPassEncoder.setVertexBuffer(0, particleVerticesBuffer);
        renderPassEncoder.setVertexBuffer(1, particleInstancesBuffer);

        // index draw
        renderPassEncoder.setIndexBuffer(quadIndicesBuffer, 'uint16');
        renderPassEncoder.drawIndexed(quadIndexArray.length, parameters.instanceNum);

        renderPassEncoder.end();
        gDevice.queue.submit([renderCommandEncoder.finish()]);

        // update と draw の queue完了を待つ
        // queryでtimestampを確認する方が丁寧
        await gDevice.queue.onSubmittedWorkDone();

        needsUpdateDirtyFlag = true;

        if(beginResetLastTime > requestResetLastTime) {
            needsResetDirtyFlag = false;
        }
    };

    const tick = (time) => {
        const immediateRaf = currentTime < 0;

        deltaTime = time / 1000 - currentTime;
        currentTime = time / 1000;

        if (immediateRaf) {
            requestAnimationFrame(tick);
            return;
        }

        if (needsUpdateDirtyFlag) {
            // このデモでは更新と描画を分けない
            update();
        }

        requestAnimationFrame(tick);
    };

    const resize = () => {
        width = wrapperElement.offsetWidth;
        height = wrapperElement.offsetHeight;
        const ratio = Math.min(1.5, window.devicePixelRatio);
        canvas.width = Math.floor(width * ratio);
        canvas.height = Math.floor(height * ratio);

        let aspect = canvas.width / canvas.height;
        const projectionMatrix = mat4.perspective((60 * Math.PI) / 180, aspect, 1, 100);
        gDevice.queue.writeBuffer(
            particleUniformBuffer,
            4 * 16 * 0,
            projectionMatrix.buffer,
            projectionMatrix.byteOffset,
            projectionMatrix.byteLength
        );
    };

    const updateViewMatrix = (nx, ny) => {
        vec3.set(nx * 10, ny * 10, -50, cameraPosition);
        const viewMatrix = mat4.lookAt(
            [cameraPosition[0], cameraPosition[1], cameraPosition[2]],
            [0, 0, 0],
            [0, 1, 0]
        );
        gDevice.queue.writeBuffer(
            particleUniformBuffer,
            4 * 16 * 1,
            viewMatrix.buffer,
            viewMatrix.byteOffset,
            viewMatrix.byteLength
        );
    }

    const mouseMove = (event) => {
        const x = event.clientX / width * 2 - 1;
        const y = event.clientY / height * 2 - 1;
        updateViewMatrix(x, y);
    };

    const initDebugger = () => {
        const pane = new Pane();
        pane.addBinding(parameters, 'instanceNum', {min: 1, max: maxInstanceNum, step: 1});
        pane.addBinding(parameters, 'color');
        pane.addBinding(parameters, 'speed', {min: 0.001, max: 10});
        pane.addBinding(parameters, 'noiseScale', {min: 0.01, max: 0.1, step: 0.001});
        pane.addBinding(parameters, 'particleScale', {min: 0.01, max: 0.5, step: 0.001});
        pane.addBinding(parameters, 'distanceFadePower', {min: 0.0001, max: 0.1, step: 0.0001});
        pane.addButton({
            title: 'Start',
        }).on('click', () => {
            needsResetDirtyFlag = true;
            requestResetLastTime = performance.now();
        });
    };

    initDebugger();

    updateViewMatrix(0, 0);

    resize();
    window.addEventListener('resize', resize);

    window.addEventListener('mousemove', mouseMove);

    requestAnimationFrame(tick);
}
main();