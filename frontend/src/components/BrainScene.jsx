import { Suspense, useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Float, MeshDistortMaterial, Points, PointMaterial } from "@react-three/drei";
import * as THREE from "three";

// Scattered points on a spherical shell, representing scan/neuron data.
function NeuronField({ count = 1400, radius = 2.6 }) {
  const ref = useRef(null);

  const positions = useMemo(() => {
    const pts = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const r = radius * (0.72 + Math.random() * 0.4);
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      pts[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      pts[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      pts[i * 3 + 2] = r * Math.cos(phi);
    }
    return pts;
  }, [count, radius]);

  useFrame((_, delta) => {
    if (ref.current) {
      ref.current.rotation.y += delta * 0.045;
      ref.current.rotation.x += delta * 0.012;
    }
  });

  return (
    <Points ref={ref} positions={positions} stride={3} frustumCulled>
      <PointMaterial
        transparent
        color="#c99a5b"
        size={0.028}
        sizeAttenuation
        depthWrite={false}
        opacity={0.75}
      />
    </Points>
  );
}

function DistortedCore() {
  const ref = useRef(null);
  useFrame((_, delta) => {
    if (ref.current) {
      ref.current.rotation.y += delta * 0.09;
      ref.current.rotation.z += delta * 0.03;
    }
  });

  return (
    <mesh ref={ref}>
      <icosahedronGeometry args={[1.55, 12]} />
      <MeshDistortMaterial
        color="#4a4f86"
        speed={1.4}
        distort={0.35}
        radius={1}
        roughness={0.35}
        metalness={0.25}
        emissive="#2a2d5c"
        emissiveIntensity={0.35}
      />
    </mesh>
  );
}

function WireframeShell() {
  const ref = useRef(null);
  useFrame((_, delta) => {
    if (ref.current) ref.current.rotation.y -= delta * 0.02;
  });
  return (
    <mesh ref={ref}>
      <icosahedronGeometry args={[2.05, 1]} />
      <meshBasicMaterial color="#8d8fc9" wireframe transparent opacity={0.14} />
    </mesh>
  );
}

function Scene() {
  return (
    <>
      <ambientLight intensity={0.55} />
      <pointLight position={[4, 3, 4]} intensity={45} color="#e8d3b0" />
      <pointLight position={[-4, -2, -3]} intensity={25} color="#5b62b8" />

      <Float speed={1.1} rotationIntensity={0.35} floatIntensity={0.6}>
        <DistortedCore />
      </Float>
      <WireframeShell />
      <NeuronField />
    </>
  );
}

export default function BrainScene({ className }) {
  return (
    <div className={className} aria-hidden="true" style={{ width: "100%", height: "100%" }}>
      <Canvas
        camera={{ position: [0, 0, 6.2], fov: 42 }}
        gl={{ antialias: true, alpha: true }}
        dpr={[1, 1.75]}
      >
        <Suspense fallback={null}>
          <Scene />
        </Suspense>
      </Canvas>
    </div>
  );
}
