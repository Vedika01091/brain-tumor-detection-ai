import { motion } from "framer-motion";
import BrainScene from "./BrainScene";

const fadeUp = {
  hidden: { opacity: 0, y: 18 },
  show: (i = 0) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.6, delay: i * 0.08, ease: [0.22, 1, 0.36, 1] },
  }),
};

export default function Hero({ accuracy }) {
  function scrollToWorkspace() {
    document.getElementById("workspace")?.scrollIntoView({ behavior: "smooth" });
  }

  return (
    <>
      <nav className="site-nav container">
        <div className="brand">
          <span className="brand-mark" role="img" aria-label="Brain">
            🧠
          </span>
          Brain Tumor Detection
        </div>
        <span className="disclaimer-badge">
          ⚠ Research prototype for academic demonstration &mdash; not a medical device.
        </span>
      </nav>

      <section className="hero container">
        <div className="hero-copy">
          <motion.span
            className="eyebrow"
            variants={fadeUp}
            initial="hidden"
            animate="show"
            custom={0}
          >
            EfficientNet-B0 Transfer Learning &middot; Grad-CAM Explainability
          </motion.span>

          <motion.h1 variants={fadeUp} initial="hidden" animate="show" custom={1}>
            See what the model sees in <em>every MRI scan</em>
          </motion.h1>

          <motion.p className="hero-sub" variants={fadeUp} initial="hidden" animate="show" custom={2}>
            Upload a brain MRI and get an instant glioma, meningioma, pituitary or
            no-tumour classification &mdash; with a Grad-CAM heatmap showing exactly
            which regions drove the prediction.
          </motion.p>

          <motion.div
            className="hero-actions"
            variants={fadeUp}
            initial="hidden"
            animate="show"
            custom={3}
          >
            <button type="button" className="btn-hero" onClick={scrollToWorkspace}>
              Analyze a Scan ↓
            </button>
            <a
              className="btn-ghost"
              href="https://github.com"
              onClick={(e) => e.preventDefault()}
            >
              How the pipeline works
            </a>
          </motion.div>

          <motion.div
            className="hero-stats"
            variants={fadeUp}
            initial="hidden"
            animate="show"
            custom={4}
          >
            <div className="hero-stat">
              <div className="value">
                {typeof accuracy === "number" ? `${(accuracy * 100).toFixed(1)}%` : "—"}
              </div>
              <div className="label">Test-set accuracy</div>
            </div>
            <div className="hero-stat">
              <div className="value">4</div>
              <div className="label">Tumour classes</div>
            </div>
            <div className="hero-stat">
              <div className="value">&lt;1s</div>
              <div className="label">Inference + heatmap</div>
            </div>
          </motion.div>
        </div>

        <div className="hero-canvas">
          <div className="hero-canvas-mask">
            <BrainScene className="hero-canvas-inner" />
          </div>
        </div>
      </section>
    </>
  );
}
