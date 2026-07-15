import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

// Self-hosted variable fonts — no runtime dependency on Google's CDN.
import "@fontsource-variable/fraunces";
import "@fontsource-variable/hanken-grotesk";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/500.css";

import "./index.css";
import App from "./App";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
