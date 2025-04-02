import { useEffect, useRef, useState } from "react";

export default function SOCkJawSwarmSim() {
  const canvasRef = useRef(null);
  const [swarm, setSwarm] = useState([]);

  const width = 800;
  const height = 600;
  const swarmSize = 35;

  useEffect(() => {
    const ctx = canvasRef.current.getContext("2d");

    // Create initial swarm
    let jawz = Array.from({ length: swarmSize }).map(() => ({
      x: Math.random() * width,
      y: Math.random() * height,
      dx: (Math.random() - 0.5) * 3,
      dy: (Math.random() - 0.5) * 3,
      hue: Math.floor(Math.random() * 360),
    }));

    const update = () => {
      ctx.clearRect(0, 0, width, height);

      for (let i = 0; i < jawz.length; i++) {
        const j = jawz[i];

        // Swarm AI logic: Avoid edges, attract to center, wander
        const cx = width / 2;
        const cy = height / 2;
        j.dx += (cx - j.x) * 0.0004 + (Math.random() - 0.5) * 0.1;
        j.dy += (cy - j.y) * 0.0004 + (Math.random() - 0.5) * 0.1;

        j.x += j.dx;
        j.y += j.dy;

        if (j.x < 0 || j.x > width) j.dx *= -1;
        if (j.y < 0 || j.y > height) j.dy *= -1;

        // Draw SOCkJaw
        ctx.beginPath();
        ctx.arc(j.x, j.y, 8, 0, 2 * Math.PI);
        ctx.fillStyle = `hsl(${j.hue}, 100%, 65%)`;
        ctx.fill();
      }

      requestAnimationFrame(update);
    };

    update();
    setSwarm(jawz);
  }, []);

  return (
    <div className="p-4 space-y-4">
      <h2 className="text-xl font-bold"><SOCkJaw Swarm Simulation</h2>
      <canvas ref={canvasRef} width={width} height={height} className="border rounded shadow-lg bg-black" />
      <p className="text-gray-600 text-sm">
        Watch as each SOCkJaw follows swarm dynamics, randomly pulsing through AI-driven behavior routines.
      </p>
    </div>
  );
}
