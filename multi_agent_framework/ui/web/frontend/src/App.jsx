import { useEffect, useRef, useState } from "react";

const API_BASE =
  import.meta.env.VITE_API_BASE ||
  `http://${window.location.hostname}:8000`;

const STATUS_LABELS = {
  pending: "pending",
  running: "running",
  completed: "done",
  failed: "failed"
};

const formatTimestamp = (value) => {
  if (!value) {
    return "";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  return date.toLocaleString();
};

const summarizePayload = (payload, limit = 160) => {
  if (!payload) {
    return "";
  }
  const text = JSON.stringify(payload);
  if (text.length <= limit) {
    return text;
  }
  return `${text.slice(0, limit)}...`;
};

function App() {
  const [goalInput, setGoalInput] = useState("");
  const [sessionGoal, setSessionGoal] = useState("");
  const [plan, setPlan] = useState([]);
  const [taskStatus, setTaskStatus] = useState({});
  const [logs, setLogs] = useState([]);
  const [running, setRunning] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [activeAgent, setActiveAgent] = useState("");
  const [activeTool, setActiveTool] = useState("");
  const [activeModel, setActiveModel] = useState("");
  const [connection, setConnection] = useState("connecting");
  const [lastResult, setLastResult] = useState(null);
  const [useLangGraph, setUseLangGraph] = useState(true);
  const [newSessionOnLaunch, setNewSessionOnLaunch] = useState(false);
  const [intentAnalysis, setIntentAnalysis] = useState(null);
  const [taskBriefs, setTaskBriefs] = useState(null);
  const [finalReport, setFinalReport] = useState(null);
  const [traceEvents, setTraceEvents] = useState([]);
  const [sessionId, setSessionId] = useState("");
  const [sessionDir, setSessionDir] = useState("");
  const [sessionList, setSessionList] = useState([]);
  const [sessionLabel, setSessionLabel] = useState("");
  const [sessionNote, setSessionNote] = useState("");
  const [artifacts, setArtifacts] = useState([]);
  const [artifactPath, setArtifactPath] = useState("");
  const [artifactContent, setArtifactContent] = useState("");
  const [artifactError, setArtifactError] = useState("");
  const [modelsInfo, setModelsInfo] = useState(null);
  const [modelsError, setModelsError] = useState("");
  const [modelTarget, setModelTarget] = useState("");
  const [modelSwitchNote, setModelSwitchNote] = useState("");
  const [modelPersist, setModelPersist] = useState(false);
  const [modelWarm, setModelWarm] = useState(true);
  const [toolPackage, setToolPackage] = useState("");
  const [toolContainer, setToolContainer] = useState("sandbox");
  const [toolInstallType, setToolInstallType] = useState("pip");
  const [toolOutput, setToolOutput] = useState("");
  const [tests, setTests] = useState([]);
  const [testName, setTestName] = useState("");
  const [testPrompt, setTestPrompt] = useState("");
  const logsEndRef = useRef(null);

  const appendLog = (line) => {
    if (!line) {
      return;
    }
    setLogs((prev) => [...prev, line].slice(-200));
  };

  const applySnapshot = (snapshot) => {
    setPlan(Array.isArray(snapshot.plan) ? snapshot.plan : []);
    setTaskStatus(snapshot.task_status || {});
    setActiveAgent(snapshot.active_agent || "");
    setActiveTool(snapshot.active_tool || "");
    setActiveModel(snapshot.active_model || "");
    setIntentAnalysis(snapshot.intent_analysis || null);
    setTaskBriefs(snapshot.task_briefs || null);
    setFinalReport(snapshot.final_report || null);
    if (snapshot.session) {
      setSessionId(snapshot.session.id || "");
      setSessionDir(snapshot.session.dir || "");
    }
    if (Array.isArray(snapshot.trace)) {
      setTraceEvents(snapshot.trace.slice(-200));
    }
    if (Array.isArray(snapshot.logs)) {
      setLogs(snapshot.logs.slice(-200));
    }
  };

  const refreshSessions = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/sessions`);
      if (!response.ok) {
        setSessionNote("Failed to load sessions.");
        return [];
      }
      const data = await response.json();
      const list = Array.isArray(data.sessions) ? data.sessions : [];
      setSessionList(list);
      setSessionNote("");
      return list;
    } catch (error) {
      setSessionNote("Failed to load sessions.");
      return [];
    }
  };

  const refreshArtifacts = async (targetSessionId) => {
    const query = targetSessionId
      ? `?session_id=${encodeURIComponent(targetSessionId)}`
      : "";
    try {
      const response = await fetch(`${API_BASE}/api/artifacts/tree${query}`);
      const data = await response.json().catch(() => ({}));
      if (!response.ok || data.ok === false) {
        setArtifacts([]);
        setArtifactError(data.error || "No artifacts available.");
        return;
      }
      setArtifacts(Array.isArray(data.entries) ? data.entries : []);
      setArtifactError("");
      if (data.session_id) {
        setSessionId(data.session_id);
      }
      if (data.session_dir) {
        setSessionDir(data.session_dir);
      }
    } catch (error) {
      setArtifacts([]);
      setArtifactError("Failed to load artifacts.");
    }
  };

  const loadArtifact = async (path) => {
    if (!path) {
      return;
    }
    setArtifactPath(path);
    setArtifactContent("");
    setArtifactError("");
    const query = sessionId
      ? `?path=${encodeURIComponent(path)}&session_id=${encodeURIComponent(sessionId)}`
      : `?path=${encodeURIComponent(path)}`;
    try {
      const response = await fetch(`${API_BASE}/api/artifacts/file${query}`);
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.ok) {
        setArtifactError(data.error || "Unable to read artifact.");
        return;
      }
      setArtifactContent(data.content || "");
    } catch (error) {
      setArtifactError("Unable to read artifact.");
    }
  };

  const refreshModels = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/models`);
      if (!response.ok) {
        setModelsError("Failed to load models.");
        return;
      }
      const data = await response.json();
      setModelsInfo(data);
      setModelsError("");
      if (data.active_model) {
        setActiveModel(data.active_model);
      }
      const available = Array.isArray(data.available) ? data.available : [];
      setModelTarget((prev) => {
        if (prev && available.includes(prev)) {
          return prev;
        }
        return data.active_model || available[0] || "";
      });
    } catch (error) {
      setModelsError("Failed to load models.");
    }
  };

  const refreshTests = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/tests`);
      if (!response.ok) {
        setTests([]);
        return;
      }
      const data = await response.json();
      setTests(Array.isArray(data.tests) ? data.tests : []);
    } catch (error) {
      setTests([]);
    }
  };

  const handleModelSwitch = async () => {
    if (running) {
      setModelSwitchNote("Stop the run before swapping models.");
      return;
    }
    const target =
      modelTarget || activeModel || modelsInfo?.active_model || "";
    if (!target) {
      setModelSwitchNote("Select a model to swap.");
      return;
    }
    setModelSwitchNote("Switching...");
    const payload = {
      model: target,
      current_model: activeModel || modelsInfo?.active_model || "",
      unload_current: true,
      warm: modelWarm,
      persist: modelPersist
    };
    try {
      const response = await fetch(`${API_BASE}/api/models/switch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.ok) {
        setModelSwitchNote(data.error || "Model switch failed.");
        return;
      }
      const active = data.active_model || target;
      setActiveModel(active);
      setModelSwitchNote(`Active model: ${active}`);
      await refreshModels();
    } catch (error) {
      setModelSwitchNote("Model switch failed.");
    }
  };

  const handleSessionNew = async (labelOverride) => {
    if (running) {
      appendLog("[UI] Session change blocked while running.");
      return;
    }
    const label = String(labelOverride || sessionLabel || "New session").trim();
    try {
      const response = await fetch(`${API_BASE}/api/session/new`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.ok) {
        setSessionNote(data.error || "Failed to create session.");
        return;
      }
      setSessionId(data.session_id || "");
      setSessionDir(data.session_dir || "");
      setSessionLabel("");
      setSessionNote(`New session ready: ${data.session_id || label}`);
      appendLog(`[UI] New session: ${data.session_id || label}`);
      await refreshSessions();
      await refreshArtifacts(data.session_id || "");
    } catch (error) {
      setSessionNote("Failed to create session.");
    }
  };

  const handleSessionRestore = async (targetId) => {
    if (running) {
      appendLog("[UI] Session change blocked while running.");
      return;
    }
    const sessionTarget = String(targetId || "").trim();
    if (!sessionTarget) {
      setSessionNote("Session id is required.");
      return;
    }
    try {
      const response = await fetch(`${API_BASE}/api/session/restore`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionTarget })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.ok) {
        setSessionNote(data.error || "Failed to restore session.");
        return;
      }
      setSessionId(data.session_id || "");
      setSessionDir(data.session_dir || "");
      setSessionNote(`Restored session: ${data.session_id}`);
      appendLog(`[UI] Restored session: ${data.session_id}`);
      await refreshArtifacts(data.session_id || "");
    } catch (error) {
      setSessionNote("Failed to restore session.");
    }
  };

  const handleMemoryCommand = async (goal) => {
    const trimmed = goal.trim();
    if (!trimmed.toLowerCase().startsWith("/memory")) {
      return false;
    }
    const parts = trimmed.split(/\s+/);
    const action = (parts[1] || "").toLowerCase();

    if (!action || action === "list") {
      const list = await refreshSessions();
      if (list.length === 0) {
        appendLog("[UI] /memory: no sessions found.");
      } else {
        appendLog(`[UI] /memory: ${list.length} session(s).`);
        list.slice(0, 6).forEach((item) => {
          appendLog(`[UI] - ${item.id} | ${item.goal || "no goal"}`);
        });
      }
      setGoalInput("");
      return true;
    }

    if (action === "new") {
      const label = parts.slice(2).join(" ") || sessionLabel || "New session";
      await handleSessionNew(label);
      setGoalInput("");
      return true;
    }

    const target = action === "latest" ? "latest" : parts[1];
    await handleSessionRestore(target);
    setGoalInput("");
    return true;
  };

  const handleLaunch = async () => {
    const goal = goalInput.trim();
    if (!goal || running) {
      return;
    }

    if (await handleMemoryCommand(goal)) {
      return;
    }

    setRunning(true);
    setSessionGoal(goal);

    const payload = { goal, use_langgraph: useLangGraph };
    if (newSessionOnLaunch) {
      payload.new_session = true;
    } else if (sessionId) {
      payload.session_id = sessionId;
    }

    const response = await fetch(`${API_BASE}/api/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      appendLog(`[UI] Launch failed: ${data.error || response.statusText}`);
      setRunning(false);
    } else {
      setGoalInput("");
    }
  };

  const handleCancel = async () => {
    const response = await fetch(`${API_BASE}/api/cancel`, { method: "POST" });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      appendLog(`[UI] Cancel failed: ${data.error || response.statusText}`);
    }
  };

  const handleReset = async () => {
    const response = await fetch(`${API_BASE}/api/reset`, { method: "POST" });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      appendLog(`[UI] Reset failed: ${data.error || response.statusText}`);
    }
  };

  const handleToolCheck = async () => {
    const tool = toolPackage.trim();
    if (!tool) {
      setToolOutput("Tool name is required.");
      return;
    }
    setToolOutput("Checking...");
    try {
      const response = await fetch(`${API_BASE}/api/tools/check`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tool, container: toolContainer })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        setToolOutput(data.error || "Tool check failed.");
        return;
      }
      const status = data.found ? "Found" : "Not found";
      setToolOutput(`${status}: ${data.tool}\n${data.output || ""}`.trim());
    } catch (error) {
      setToolOutput("Tool check failed.");
    }
  };

  const handleToolInstall = async () => {
    const packageName = toolPackage.trim();
    if (!packageName) {
      setToolOutput("Package name is required.");
      return;
    }
    setToolOutput("Installing...");
    try {
      const response = await fetch(`${API_BASE}/api/tools/install`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          package: packageName,
          container: toolContainer,
          install_type: toolInstallType
        })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.ok) {
        setToolOutput(data.output || data.error || "Install failed.");
        return;
      }
      setToolOutput(data.output || "Install complete.");
    } catch (error) {
      setToolOutput("Install failed.");
    }
  };

  const handleTestAdd = async () => {
    const prompt = testPrompt.trim();
    if (!prompt) {
      appendLog("[UI] Test prompt is required.");
      return;
    }
    try {
      const response = await fetch(`${API_BASE}/api/tests`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: testName.trim(), prompt })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.ok) {
        appendLog(`[UI] Test add failed: ${data.error || "unknown error"}`);
        return;
      }
      setTests(Array.isArray(data.tests) ? data.tests : []);
      setTestName("");
      setTestPrompt("");
    } catch (error) {
      appendLog("[UI] Test add failed.");
    }
  };

  const handleTestDelete = async (testId) => {
    try {
      const response = await fetch(`${API_BASE}/api/tests/${testId}`, {
        method: "DELETE"
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.ok) {
        appendLog(`[UI] Test delete failed: ${data.error || "unknown error"}`);
        return;
      }
      setTests(Array.isArray(data.tests) ? data.tests : []);
    } catch (error) {
      appendLog("[UI] Test delete failed.");
    }
  };

  const handleTestRun = async (prompt) => {
    if (!prompt) {
      return;
    }
    if (running) {
      appendLog("[UI] Run in progress. Wait for completion.");
      return;
    }
    setRunning(true);
    setSessionGoal(prompt);

    const payload = { goal: prompt, use_langgraph: useLangGraph };
    if (newSessionOnLaunch) {
      payload.new_session = true;
    } else if (sessionId) {
      payload.session_id = sessionId;
    }

    const response = await fetch(`${API_BASE}/api/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      appendLog(`[UI] Test run failed: ${data.error || response.statusText}`);
      setRunning(false);
    }
  };

  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/state`);
        const data = await res.json();
        setRunning(Boolean(data.running));
        setCancelling(Boolean(data.cancelling));
        setSessionGoal(data.goal || "");
        setUseLangGraph(
          data.use_langgraph === undefined ? true : Boolean(data.use_langgraph)
        );
        const sessionFromApi =
          data.ui?.session?.id || data.session_id || "";
        const sessionDirFromApi =
          data.ui?.session?.dir || data.session_dir || "";
        setSessionId(sessionFromApi);
        setSessionDir(sessionDirFromApi);
        if (data.ui) {
          applySnapshot(data.ui);
        }
        if (data.last_result) {
          setLastResult(data.last_result);
        }
        await refreshSessions();
        await refreshModels();
        await refreshTests();
        if (sessionFromApi) {
          await refreshArtifacts(sessionFromApi);
        }
      } catch (error) {
        setConnection("error");
      }
    };
    load();
  }, []);

  useEffect(() => {
    const source = new EventSource(`${API_BASE}/api/events`);
    setConnection("connecting");

    source.onopen = () => setConnection("connected");
    source.onerror = () => setConnection("error");

    source.addEventListener("snapshot", (event) => {
      const data = JSON.parse(event.data);
      applySnapshot(data);
    });

    source.addEventListener("plan", (event) => {
      const data = JSON.parse(event.data);
      setPlan(Array.isArray(data.plan) ? data.plan : []);
    });

    source.addEventListener("task_status", (event) => {
      const data = JSON.parse(event.data);
      setTaskStatus((prev) => ({
        ...prev,
        [data.task_id]: data.status
      }));
    });

    source.addEventListener("log", (event) => {
      const data = JSON.parse(event.data);
      appendLog(data.line || data.message);
    });

    source.addEventListener("session_start", (event) => {
      const data = JSON.parse(event.data);
      setRunning(true);
      setSessionGoal(data.goal || "");
    });

    source.addEventListener("session_end", () => {
      setRunning(false);
      setCancelling(false);
    });

    source.addEventListener("run_start", (event) => {
      const data = JSON.parse(event.data);
      setRunning(true);
      setCancelling(false);
      setSessionGoal(data.goal || "");
    });

    source.addEventListener("run_result", (event) => {
      const data = JSON.parse(event.data);
      setLastResult(data.result || null);
      setRunning(false);
      setCancelling(false);
      refreshArtifacts();
    });

    source.addEventListener("run_cancelling", () => {
      setCancelling(true);
    });

    source.addEventListener("run_cancelled", () => {
      setRunning(false);
      setCancelling(false);
    });

    source.addEventListener("active_agent", (event) => {
      const data = JSON.parse(event.data);
      setActiveAgent(data.agent || "");
    });

    source.addEventListener("active_tool", (event) => {
      const data = JSON.parse(event.data);
      setActiveTool(data.tool || "");
    });

    source.addEventListener("active_model", (event) => {
      const data = JSON.parse(event.data);
      setActiveModel(data.model || "");
    });

    source.addEventListener("session", (event) => {
      const data = JSON.parse(event.data);
      setSessionId(data.session_id || "");
      setSessionDir(data.session_dir || "");
      refreshSessions();
      refreshArtifacts(data.session_id || "");
    });

    source.addEventListener("trace", (event) => {
      const data = JSON.parse(event.data);
      const entry = data.event || data;
      setTraceEvents((prev) => [...prev, entry].slice(-200));
    });

    source.addEventListener("intent_analysis", (event) => {
      const data = JSON.parse(event.data);
      setIntentAnalysis(data.payload || data);
    });

    source.addEventListener("task_briefs", (event) => {
      const data = JSON.parse(event.data);
      setTaskBriefs(data.payload || data);
    });

    source.addEventListener("final_report", (event) => {
      const data = JSON.parse(event.data);
      setFinalReport(data.payload || data);
    });

    source.addEventListener("reset", () => {
      setPlan([]);
      setTaskStatus({});
      setActiveAgent("");
      setActiveTool("");
      setActiveModel("");
      setModelSwitchNote("");
      setIntentAnalysis(null);
      setTaskBriefs(null);
      setFinalReport(null);
      setTraceEvents([]);
      setLogs([]);
      setSessionGoal("");
      setLastResult(null);
      setRunning(false);
      setCancelling(false);
      setUseLangGraph(true);
    });

    return () => source.close();
  }, []);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const statusLabel = cancelling ? "Cancelling" : running ? "Running" : "Idle";

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="brand-mark" />
          <div>
            <div className="brand-title">AI Autonom</div>
            <div className="brand-sub">Multi-agent orchestration control</div>
          </div>
        </div>
        <div
          className={`status-pill ${cancelling ? "cancelling" : running ? "running" : "idle"}`}
        >
          <span className="status-dot" />
          <span>{statusLabel}</span>
          <span className="status-conn">{connection}</span>
        </div>
      </header>

      <main className="grid">
        <section className="panel mission" style={{ "--delay": "0s" }}>
          <div className="panel-header">
            <h2>Mission Control</h2>
            <span className="panel-chip">Primary</span>
          </div>
          <p className="panel-copy">
            Define the objective, watch the orchestration, and track handoffs in real time.
          </p>
          <div className="input-stack">
            <label htmlFor="goal">Goal</label>
            <textarea
              id="goal"
              value={goalInput}
              onChange={(event) => setGoalInput(event.target.value)}
              placeholder="Ship a multi-agent workflow with memory, tools, and a clear plan."
              rows={4}
            />
            <div className="input-actions">
              <button className="primary" onClick={handleLaunch} disabled={running}>
                {running ? "Running" : "Launch"}
              </button>
              <button className="ghost" onClick={handleCancel} disabled={!running || cancelling}>
                {cancelling ? "Cancelling" : "Cancel"}
              </button>
              <button className="secondary" onClick={handleReset} disabled={running}>
                Reset
              </button>
              <div className="input-meta">
                <span>Active agent: {activeAgent || "-"}</span>
                <span>Active tool: {activeTool || "-"}</span>
                <span>Active model: {activeModel || "-"}</span>
              </div>
            </div>
            <div className="toggle-row">
              <label className="switch">
                <input
                  type="checkbox"
                  checked={useLangGraph}
                  onChange={(event) => setUseLangGraph(event.target.checked)}
                />
                <span className="slider" />
              </label>
              <span className="toggle-label">Use LangGraph pipeline</span>
            </div>
            <div className="toggle-row">
              <label className="switch">
                <input
                  type="checkbox"
                  checked={newSessionOnLaunch}
                  onChange={(event) => setNewSessionOnLaunch(event.target.checked)}
                />
                <span className="slider" />
              </label>
              <span className="toggle-label">Start new session on launch</span>
            </div>
          </div>
          <div className="mission-meta">
            <div>
              <span className="meta-label">Session goal</span>
              <span className="meta-value">{sessionGoal || "None"}</span>
            </div>
            <div>
              <span className="meta-label">Session id</span>
              <span className="meta-value">{sessionId || "None"}</span>
            </div>
            <div>
              <span className="meta-label">Tasks tracked</span>
              <span className="meta-value">{plan.length}</span>
            </div>
          </div>
          <div className="mission-hint">
            Commands: /memory list | /memory latest | /memory &lt;session_id&gt;
          </div>
        </section>

        <section className="panel feed" style={{ "--delay": "0.08s" }}>
          <div className="panel-header">
            <h2>Live Feed</h2>
            <span className="panel-chip">Stream</span>
          </div>
          <div className="feed-order">
            <div className="feed-order-item">
              <div className="feed-order-title">1. Intent Analysis (JSON)</div>
              <pre className="feed-json">
                {intentAnalysis
                  ? JSON.stringify(intentAnalysis, null, 2)
                  : "Waiting for intent analysis..."}
              </pre>
            </div>
            <div className="feed-order-item">
              <div className="feed-order-title">2. Task Briefs (Tools & Models)</div>
              <pre className="feed-json">
                {taskBriefs
                  ? JSON.stringify(taskBriefs, null, 2)
                  : "Waiting for task briefs..."}
              </pre>
            </div>
          </div>
          <div className="feed-divider" />
          <div className="feed-order-title">3. Execution Log</div>
          <div className="log-window">
            {logs.length === 0 ? (
              <div className="log-empty">Waiting for orchestrator events.</div>
            ) : (
              logs.map((line, index) => (
                <div className="log-line" key={`${index}-${line.slice(0, 8)}`}>
                  {line}
                </div>
              ))
            )}
            <div ref={logsEndRef} />
          </div>
          <div className="feed-divider" />
          <div className="feed-order-item">
            <div className="feed-order-title">4. Final Report</div>
            <div className="feed-sub">
              {finalReport?.path ? `Saved to ${finalReport.path}` : "No report yet."}
            </div>
            <pre className="feed-json">
              {finalReport?.content || "Waiting for final report..."}
            </pre>
          </div>
        </section>

        <section className="panel session" style={{ "--delay": "0.16s" }}>
          <div className="panel-header">
            <h2>Session Memory</h2>
            <span className="panel-chip">Memory</span>
          </div>
          <div className="session-current">
            <div>
              <span className="meta-label">Current session</span>
              <span className="meta-value">{sessionId || "None"}</span>
            </div>
            <div>
              <span className="meta-label">Session dir</span>
              <span className="meta-value">{sessionDir || "Not set"}</span>
            </div>
          </div>
          <div className="session-actions">
            <input
              className="control-input"
              value={sessionLabel}
              onChange={(event) => setSessionLabel(event.target.value)}
              placeholder="New session label"
            />
            <button className="btn secondary" onClick={() => handleSessionNew()}>
              New session
            </button>
            <button className="btn ghost" onClick={refreshSessions}>
              Refresh
            </button>
          </div>
          <div className="session-hint">
            Use /memory latest or pick a session below to restore.
          </div>
          <div className="session-list">
            {sessionList.length === 0 ? (
              <div className="session-empty">No sessions yet.</div>
            ) : (
              sessionList.map((item) => (
                <div
                  key={item.id}
                  className={`session-item ${item.id === sessionId ? "active" : ""}`}
                >
                  <div>
                    <div className="session-title">{item.id}</div>
                    <div className="session-sub">
                      {item.goal || "no goal"} | {formatTimestamp(item.created_at)}
                    </div>
                  </div>
                  <button
                    className="btn ghost small"
                    onClick={() => handleSessionRestore(item.id)}
                  >
                    Restore
                  </button>
                </div>
              ))
            )}
          </div>
          {sessionNote ? <div className="session-note">{sessionNote}</div> : null}
        </section>

        <section className="panel trace" style={{ "--delay": "0.24s" }}>
          <div className="panel-header">
            <h2>Trace Timeline</h2>
            <span className="panel-chip">Trace</span>
          </div>
          <div className="trace-list">
            {traceEvents.length === 0 ? (
              <div className="trace-empty">No trace events yet.</div>
            ) : (
              traceEvents.slice(-120).map((event, index) => (
                <div className="trace-item" key={`${event.type}-${index}`}>
                  <div className="trace-meta">
                    <span className="trace-type">{event.type || "event"}</span>
                    <span className="trace-time">
                      {event.ts ? formatTimestamp(event.ts) : ""}
                    </span>
                  </div>
                  <div className="trace-payload">
                    {summarizePayload(event.payload)}
                  </div>
                </div>
              ))
            )}
          </div>
        </section>

        <section className="panel plan" style={{ "--delay": "0.32s" }}>
          <div className="panel-header">
            <h2>Execution Plan</h2>
            <span className="panel-chip">Plan</span>
          </div>
          {plan.length === 0 ? (
            <div className="plan-empty">No plan yet. Launch a run to generate tasks.</div>
          ) : (
            <div className="plan-list">
              {plan.map((task) => {
                const taskId = task.id || "unknown";
                const status = taskStatus[taskId] || "pending";
                return (
                  <div className="plan-item" key={taskId}>
                    <div className={`status-tag ${status}`}>
                      {STATUS_LABELS[status] || status}
                    </div>
                    <div className="plan-body">
                      <div className="plan-title">{taskId}</div>
                      <div className="plan-sub">
                        {task.assigned_agent || "unassigned"}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </section>

        <section className="panel artifacts" style={{ "--delay": "0.4s" }}>
          <div className="panel-header">
            <h2>Artifacts Explorer</h2>
            <div className="panel-actions">
              <button className="btn ghost" onClick={() => refreshArtifacts()}>
                Refresh
              </button>
              <span className="panel-chip">Artifacts</span>
            </div>
          </div>
          <div className="artifact-grid">
            <div className="artifact-list">
              {artifacts.length === 0 ? (
                <div className="artifact-empty">
                  {artifactError || "No artifacts yet."}
                </div>
              ) : (
                artifacts.map((item) => (
                  <button
                    key={item.path}
                    className={`artifact-item ${artifactPath === item.path ? "active" : ""}`}
                    onClick={() => loadArtifact(item.path)}
                  >
                    <span className="artifact-path">{item.path}</span>
                    <span className="artifact-size">{item.size} bytes</span>
                  </button>
                ))
              )}
            </div>
            <div className="artifact-preview">
              <div className="artifact-preview-title">
                {artifactPath || "Select a file to preview"}
              </div>
              {artifactError ? (
                <div className="artifact-error">{artifactError}</div>
              ) : (
                <pre className="artifact-content">
                  {artifactContent || "No file selected."}
                </pre>
              )}
            </div>
          </div>
        </section>

        <section className="panel models" style={{ "--delay": "0.48s" }}>
          <div className="panel-header">
            <h2>Model Manager</h2>
            <div className="panel-actions">
              <button className="btn ghost" onClick={refreshModels}>
                Refresh
              </button>
              <span className="panel-chip">Models</span>
            </div>
          </div>
          <div className="model-active">
            Active model: {activeModel || modelsInfo?.active_model || "-"}
          </div>
          <div className="model-switch">
            <label>Target model</label>
            <select
              className="control-select"
              value={modelTarget}
              onChange={(event) => setModelTarget(event.target.value)}
            >
              <option value="">Select model</option>
              {(modelsInfo?.available || []).map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
            <div className="model-switch-actions">
              <button className="btn primary" onClick={handleModelSwitch}>
                Swap model
              </button>
              <label className="check">
                <input
                  type="checkbox"
                  checked={modelPersist}
                  onChange={(event) => setModelPersist(event.target.checked)}
                />
                Persist default
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={modelWarm}
                  onChange={(event) => setModelWarm(event.target.checked)}
                />
                Warm model
              </label>
            </div>
            {modelSwitchNote ? (
              <div className="model-switch-note">{modelSwitchNote}</div>
            ) : null}
          </div>
          {modelsError ? <div className="model-error">{modelsError}</div> : null}
          <div className="model-group">
            <div className="model-label">Available</div>
            <div className="tag-list">
              {modelsInfo?.available?.length ? (
                modelsInfo.available.map((name) => (
                  <span className="tag" key={name}>
                    {name}
                  </span>
                ))
              ) : (
                <span className="tag muted">No models found</span>
              )}
            </div>
          </div>
          <div className="model-grid">
            <div className="model-card">
              <div className="model-label">Coding</div>
              <div className="tag-list">
                {modelsInfo?.config?.coding_models?.length ? (
                  modelsInfo.config.coding_models.map((name) => (
                    <span className="tag" key={name}>
                      {name}
                    </span>
                  ))
                ) : (
                  <span className="tag muted">None set</span>
                )}
              </div>
            </div>
            <div className="model-card">
              <div className="model-label">Reasoning</div>
              <div className="tag-list">
                {modelsInfo?.config?.reasoning_models?.length ? (
                  modelsInfo.config.reasoning_models.map((name) => (
                    <span className="tag" key={name}>
                      {name}
                    </span>
                  ))
                ) : (
                  <span className="tag muted">None set</span>
                )}
              </div>
            </div>
            <div className="model-card">
              <div className="model-label">Linguistic</div>
              <div className="tag-list">
                {modelsInfo?.config?.linguistic_models?.length ? (
                  modelsInfo.config.linguistic_models.map((name) => (
                    <span className="tag" key={name}>
                      {name}
                    </span>
                  ))
                ) : (
                  <span className="tag muted">None set</span>
                )}
              </div>
            </div>
          </div>
          <div className="model-group">
            <div className="model-label">Rankings</div>
            <div className="tag-list">
              {modelsInfo?.rankings?.length ? (
                modelsInfo.rankings.slice(0, 5).map((item) => (
                  <span className="tag" key={`${item.rank}-${item.model_name}`}>
                    {item.rank}. {item.model_name}
                  </span>
                ))
              ) : (
                <span className="tag muted">No rankings</span>
              )}
            </div>
          </div>
          <div className="model-note">
            VRAM limit: {modelsInfo?.config?.vram_limit_gb ?? "unknown"} GB
          </div>
        </section>

        <section className="panel tools" style={{ "--delay": "0.56s" }}>
          <div className="panel-header">
            <h2>Tooling & Containers</h2>
            <span className="panel-chip">Tools</span>
          </div>
          <div className="tool-form">
            <div className="tool-row">
              <label>Package or tool name</label>
              <input
                className="control-input"
                value={toolPackage}
                onChange={(event) => setToolPackage(event.target.value)}
                placeholder="clang, ripgrep, pytest"
              />
            </div>
            <div className="tool-row">
              <label>Container</label>
              <select
                className="control-select"
                value={toolContainer}
                onChange={(event) => setToolContainer(event.target.value)}
              >
                <option value="sandbox">sandbox</option>
                <option value="analysis">analysis</option>
                <option value="web">web</option>
                <option value="security">security</option>
              </select>
            </div>
            <div className="tool-row">
              <label>Install type</label>
              <select
                className="control-select"
                value={toolInstallType}
                onChange={(event) => setToolInstallType(event.target.value)}
              >
                <option value="pip">pip</option>
                <option value="apt">apt</option>
              </select>
            </div>
            <div className="tool-actions">
              <button className="btn secondary" onClick={handleToolCheck}>
                Check
              </button>
              <button className="btn primary" onClick={handleToolInstall}>
                Install
              </button>
            </div>
          </div>
          <pre className="tool-output">{toolOutput || "No tool output yet."}</pre>
        </section>

        <section className="panel tests" style={{ "--delay": "0.64s" }}>
          <div className="panel-header">
            <h2>Regression Harness</h2>
            <div className="panel-actions">
              <button className="btn ghost" onClick={refreshTests}>
                Refresh
              </button>
              <span className="panel-chip">Tests</span>
            </div>
          </div>
          <div className="tests-form">
            <input
              className="control-input"
              value={testName}
              onChange={(event) => setTestName(event.target.value)}
              placeholder="Test name (optional)"
            />
            <textarea
              className="control-textarea"
              value={testPrompt}
              onChange={(event) => setTestPrompt(event.target.value)}
              placeholder="Test prompt"
              rows={3}
            />
            <button className="btn primary" onClick={handleTestAdd}>
              Save test
            </button>
          </div>
          <div className="test-list">
            {tests.length === 0 ? (
              <div className="test-empty">No saved tests yet.</div>
            ) : (
              tests.map((test) => (
                <div className="test-item" key={test.id}>
                  <div className="test-title">{test.name || test.id}</div>
                  <div className="test-prompt">{test.prompt}</div>
                  <div className="test-actions">
                    <button
                      className="btn secondary small"
                      onClick={() => handleTestRun(test.prompt)}
                    >
                      Run
                    </button>
                    <button
                      className="btn ghost small"
                      onClick={() => handleTestDelete(test.id)}
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </section>

        <section className="panel system" style={{ "--delay": "0.72s" }}>
          <div className="panel-header">
            <h2>System Snapshot</h2>
            <span className="panel-chip">State</span>
          </div>
          <div className="system-grid">
            <div>
              <span className="meta-label">Connection</span>
              <span className="meta-value">{connection}</span>
            </div>
            <div>
              <span className="meta-label">Runtime</span>
              <span className="meta-value">{running ? "Busy" : "Ready"}</span>
            </div>
            <div>
              <span className="meta-label">Active agent</span>
              <span className="meta-value">{activeAgent || "-"}</span>
            </div>
            <div>
              <span className="meta-label">Active tool</span>
              <span className="meta-value">{activeTool || "-"}</span>
            </div>
            <div>
              <span className="meta-label">Active model</span>
              <span className="meta-value">{activeModel || "-"}</span>
            </div>
            <div>
              <span className="meta-label">Session id</span>
              <span className="meta-value">{sessionId || "-"}</span>
            </div>
          </div>
          <div className="result-block">
            <div className="result-title">Last Result</div>
            <pre>{lastResult ? JSON.stringify(lastResult, null, 2) : "No result yet."}</pre>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
