import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { ConsoleProvider } from "./lib/useConsole";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <ConsoleProvider>
        <App />
      </ConsoleProvider>
    </BrowserRouter>
  </React.StrictMode>,
);
