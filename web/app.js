const $ = id => document.getElementById(id);
const svgNS = "http://www.w3.org/2000/svg";
const colors = ["#28594c", "#b8873c"];
let state = null,
    busy = false,
    mode = "pvp",
    humanPlayer = 0,
    tool = "move",
    toastTimer = null,
    watching = false,
    watchTimer = null,
    replayData = null,
    replayIndex = 0,
    replayPlaying = false,
    replayTimer = null,
    liveState = null;
let algorithms = {
    alpha_beta: {label: "Alpha-Beta", description: "最短路径评估与对抗搜索，无需模型即可对战。"},
    mcts: {label: "MCTS", description: "通过多次模拟比较走法，无需模型即可对战。"},
    alphazero: {label: "AlphaZero", description: "需要先载入训练权重。"}
};
let difficulties = {
    easy: {label: "入门", seconds: 0.15}, normal: {label: "标准", seconds: 0.6},
    hard: {label: "困难", seconds: 1.8}, expert: {label: "挑战", seconds: 4.0}
};

function toast(message, error = false) {
    clearTimeout(toastTimer);
    $("toast").textContent = message;
    $("toast").classList.toggle("error", error);
    $("toast").hidden = false;
    toastTimer = setTimeout(() => {
        $("toast").hidden = true;
    }, error ? 8500 : 4000);
}

async function request(path, body = undefined) {
    let response;
    try {
        response = await fetch(path, body === undefined ? {} : {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(body)
        });
    } catch {
        throw new Error("无法连接本地服务，请确认 app.py 正在运行。");
    }
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "操作失败，请重试。");
    return result;
}

function setBusy(value, message = "AI 正在思考…") {
    busy = value;
    if (value) clearTimeout(watchTimer);
    $("busy-overlay").hidden = !value;
    $("busy-text").textContent = message;
    $("board-area").classList.toggle("busy", value);
    document.querySelectorAll("button,select,input[type=number]").forEach(element => {
        element.disabled = value;
    });
    if (!value) {
        $("undo").disabled = !state?.can_undo;
        $("load").disabled = !$("model-select").value;
    }
    if (value) $("turn-text").textContent = message;
    else updateStatus();
    updateControls();
    if (!value) queueWatch();
}

function playerName(player) {
    if (state?.mode === "watch" || replayData) {
        const agent = state?.players?.[player];
        if (agent?.kind === "ai") return algorithms[agent.algorithm]?.label || "AI";
        return player === 0 ? "玩家一" : "玩家二";
    }
    return state?.mode === "ai" ? (player === state.human_player ? "你" : "AI") : (player === 0 ? "玩家一" : "玩家二");
}

function updateControls() {
    const finished = state?.finished ?? state?.winner !== null;
    const watch = state?.mode === "watch" && !replayData;
    $("watch-controls").hidden = !watch;
    $("watch-toggle").disabled = !watch || finished;
    $("watch-toggle").textContent = watching ? "暂停观战" : busy ? "本步后暂停" : "继续观战";
    $("watch-step").disabled = busy || watching || finished;
    $("watch-speed").disabled = !watch;
    $("undo").disabled = busy || !state?.can_undo || !!replayData;
    $("match-actions").hidden = !!replayData;
    $("save-record").disabled = busy || !state;
    $("save-record").textContent = replayData ? "⇩ 下载棋谱" : "⇩ 保存棋谱";
    $("open-saved-record").disabled = busy || !$("saved-records").value;
    $("replay-controls").hidden = !replayData;
    const length = replayData?.record.moves.length || 0;
    for (const id of ["replay-first", "replay-prev"]) $(id).disabled = busy || !replayData || replayIndex === 0;
    for (const id of ["replay-next", "replay-last"]) $(id).disabled = busy || !replayData || replayIndex === length;
    $("replay-play").disabled = busy || !replayData || !length;
    $("replay-play").textContent = replayPlaying ? "暂停" : "播放";
    $("replay-slider").disabled = busy || !replayData || !length;
}

function updateStatus() {
    if (!state || busy) return;
    const terminal = state.finished ?? state.winner !== null;
    const name = player => state.mode === "watch" ? `${player === 0 ? "绿色" : "金色"} ${playerName(player)}` : playerName(player);
    $("turn-text").textContent = replayData ? `棋谱回放 · ${replayIndex} / ${replayData.record.moves.length} 步` :
        terminal ? state.winner === null ? "达到步数上限，本局平局" : `${name(state.winner)}获胜，抵达对岸！` :
        `${name(state.current_player)}的回合`;
    $("turn-dot").style.background = colors[state.winner ?? state.current_player];
    const engine = state.algorithm || "alphazero";
    $("match-mode").textContent = replayData ? "棋谱回放" : state.mode === "watch" ? "AI 对战" : state.mode === "ai" ?
        `${algorithms[engine]?.label || engine} · ${difficulties[state.difficulty]?.label || "标准"}` : "同屏双人";
    $("retry-ai").hidden = !!replayData || terminal || state.mode !== "ai" || state.human_turn;
    $("board-caption").textContent = replayData ? "使用下方进度条或箭头，查看每一步。" :
        terminal ? "这一局已经结束，可保存棋谱或重新开始。" : state.mode === "watch" ?
        `${watching ? "双方 AI 自动轮流落子。" : "观战已暂停，可继续或单步推进。"}上限 ${state.max_moves} 步。` : !state.human_turn ?
        "轮到 AI，正在思考。" : tool === "move" ? "点击高亮格子移动；遇到对手可以跳跃。" : "点击棋格间的交点放墙；绿色可放，红色不可放。";
    for (let player = 0; player < 2; player++) {
        $("name-" + player).textContent = playerName(player);
        $("walls-" + player).textContent = state.walls_left[player];
        const agent = state.players?.[player];
        $("player-" + player).querySelector(".player-meta").textContent = state.mode === "watch" && agent ?
            `${difficulties[agent.difficulty]?.label || ""} · 目标${player === 0 ? "向下 ↓" : "向上 ↑"}` :
            `目标：${player === 0 ? "下方底线 ↓" : "上方底线 ↑"}`;
        $("player-" + player).classList.toggle("active", terminal ? state.winner === player : state.current_player ===
            player);
    }
}

function svgElement(tag, attributes, text = null) {
    const node = document.createElementNS(svgNS, tag);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
    if (text !== null) node.textContent = text;
    return node;
}

function wallShape(direction, row, column) {
    return direction === 0 ? {
        x: 34 + column * 60,
        y: 86 + row * 60,
        width: 112,
        height: 8,
        rx: 3
    } : {
        x: 86 + column * 60,
        y: 34 + row * 60,
        width: 8,
        height: 112,
        rx: 3
    };
}

function drawBoard() {
    if (!state) return;
    const board = $("board");
    board.replaceChildren();
    const legal = new Set(state.legal_actions),
        last = state.moves.at(-1)?.action;
    for (let index = 0; index < 9; index++) {
        board.append(svgElement("text", {
            x: 60 + index * 60,
            y: 18,
            "text-anchor": "middle",
            fill: "#9aa18f",
            "font-size": 10
        }, String.fromCharCode(65 + index)));
        board.append(svgElement("text", {
            x: 18,
            y: 64 + index * 60,
            "text-anchor": "middle",
            fill: "#9aa18f",
            "font-size": 10
        }, String(index + 1)));
    }
    board.append(svgElement("line", {
        x1: 34,
        y1: 25,
        x2: 566,
        y2: 25,
        stroke: colors[1],
        "stroke-width": 3,
        "stroke-linecap": "round",
        opacity: .48
    }));
    board.append(svgElement("line", {
        x1: 34,
        y1: 575,
        x2: 566,
        y2: 575,
        stroke: colors[0],
        "stroke-width": 3,
        "stroke-linecap": "round",
        opacity: .48
    }));
    for (let row = 0; row < 9; row++)
        for (let column = 0; column < 9; column++) {
            const action = row * 9 + column,
                allowed = tool === "move" && state.human_turn && legal.has(action),
                selected = last === action;
            const cell = svgElement("rect", {
                x: 34 + column * 60,
                y: 34 + row * 60,
                width: 52,
                height: 52,
                rx: 6,
                class: `cell${allowed?" can-move":""}${selected?" last-cell":""}`,
                "data-action": action,
                role: "button",
                tabindex: allowed ? 0 : -1,
                "aria-label": `走到 ${String.fromCharCode(65+column)}${row+1}${allowed?"，合法走法":""}`
            });
            board.append(cell);
            if (allowed) board.append(svgElement("circle", {
                cx: 60 + column * 60,
                cy: 60 + row * 60,
                r: 4,
                fill: colors[state.current_player],
                opacity: .5,
                "pointer-events": "none"
            }));
        }
    state.walls.forEach((positions, direction) => positions.forEach(([row, column]) => {
        const action = 81 + direction * 64 + row * 8 + column;
        board.append(svgElement("rect", {
            ...wallShape(direction, row, column),
            class: `wall${last===action?" last-wall":""}`
        }));
    }));
    state.pawns.forEach(([row, column], player) => {
        board.append(svgElement("circle", {
            cx: 60 + column * 60,
            cy: 60 + row * 60,
            r: 17,
            fill: colors[player],
            stroke: "#fffdf3",
            "stroke-width": 3,
            class: "pawn"
        }));
        board.append(svgElement("circle", {
            cx: 56 + column * 60,
            cy: 55 + row * 60,
            r: 4,
            fill: "white",
            opacity: .22,
            "pointer-events": "none"
        }));
    });
    if (tool !== "move" && state.human_turn && state.winner === null)
        for (let row = 0; row < 8; row++)
            for (let column = 0; column < 8; column++) {
                const direction = tool === "horizontal" ? 0 : 1,
                    action = 81 + direction * 64 + row * 8 + column,
                    valid = legal.has(action),
                    x = 90 + column * 60,
                    y = 90 + row * 60;
                const group = svgElement("g", {
                    class: `wall-target${valid?"":" invalid"}`,
                    "data-action": action,
                    role: "button",
                    tabindex: valid ? 0 : -1,
                    "aria-label": `${direction===0?"横墙":"竖墙"} ${String.fromCharCode(65+column)}${row+1}${valid?"，可放置":"，不可放置"}`
                });
                group.append(svgElement("rect", {
                    ...wallShape(direction, row, column),
                    class: "wall-preview"
                }));
                group.append(svgElement("circle", {
                    cx: x,
                    cy: y,
                    r: 2.5,
                    class: "wall-marker"
                }));
                group.append(svgElement("rect", {
                    x: x - 26,
                    y: y - 26,
                    width: 52,
                    height: 52,
                    rx: 5,
                    class: "wall-hit"
                }));
                board.append(group);
            }
}

function actionLabel(action) {
    if (action < 81) return `${String.fromCharCode(65+action%9)}${Math.floor(action/9)+1}`;
    const value = action - 81,
        direction = Math.floor(value / 64),
        anchor = value % 64;
    return `${direction===0?"横":"竖"} ${String.fromCharCode(65+anchor%8)}${Math.floor(anchor/8)+1}`;
}

function render(next) {
    const newlySaved = next.record_saved && (!state?.record_saved || state.saved_record !== next.saved_record);
    state = next;
    if (!replayData && state.id) sessionStorage.setItem("quoridor-match", state.id);
    if (!replayData && state.finished) stopWatch();
    $("step-count").textContent = state.move_count;
    updateStatus();
    drawBoard();
    const list = $("move-list");
    list.replaceChildren();
    const recent = state.moves.slice(-8),
        offset = state.moves.length - recent.length;
    if (!recent.length) {
        const item = document.createElement("li");
        item.className = "empty-history";
        item.textContent = replayData ? "棋谱起始局面。" : state.mode === "watch" ? "等待 AI 第一步。" : "选择一个高亮格子，开始对局。";
        list.append(item);
    }
    recent.forEach((move, index) => {
        const item = document.createElement("li"),
            number = document.createElement("span"),
            dot = document.createElement("span"),
            label = document.createElement("span");
        number.className = "move-number";
        number.textContent = String(offset + index + 1).padStart(2, "0");
        dot.className = "move-dot";
        dot.style.background = colors[move.player];
        label.textContent = actionLabel(move.action);
        item.append(number, dot, label);
        list.append(item);
    });
    const aiMove = [...state.moves].reverse().find(move => move.seconds !== null);
    const stats = state.ai_stats;
    const work = stats?.algorithm === "alpha_beta" ? (stats.depth ? ` · 完成 ${stats.depth} 层搜索` : " · 路径评估") :
        stats?.simulations ? ` · ${stats.simulations} 次模拟` : "";
    $("ai-time").textContent = aiMove ? `思考 ${aiMove.seconds.toFixed(1)}s${work}` : "等待第一步";
    $("undo").disabled = busy || !state.can_undo;
    $("record-status").textContent = replayData ? "只读回放，返回对局后可继续游戏。" : state.record_error ||
        (state.record_saved ? `已保存到 records/${state.saved_record}` : "结束时自动保存，也可保存当前进度。");
    updateControls();
    if (newlySaved) refreshRecords().catch(() => {});
}

function selectMode(value) {
    if (state?.mode === "watch" && value !== "watch") stopWatch();
    mode = value;
    for (const entry of ["ai", "pvp", "watch"]) {
        $("mode-" + entry).classList.toggle("selected", entry === value);
        $("mode-" + entry).setAttribute("aria-pressed", String(entry === value));
    }
    $("ai-options").hidden = value !== "ai";
    $("watch-options").hidden = value !== "watch";
    $("start").textContent = value === "watch" ? "开始 AI 对战 ↗" : value === "ai" ? "开始人机对战 ↗" : "开始双人对战 ↗";
    updateAIOptions();
}

function updateAIOptions() {
    const engine = $("algorithm").value, level = $("difficulty").value;
    $("neural-options").hidden = engine !== "alphazero";
    if (mode === "ai" && engine === "alphazero") $("model-panel").open = true;
    $("algorithm-hint").textContent = engine === "alphazero" ?
        "使用训练权重，棋力取决于模型与搜索次数。" :
        `无需模型 · 每步最多思考约 ${difficulties[level].seconds} 秒。`;
    $("mode-hint").textContent = mode === "watch" ? "双方独立设置。AlphaZero 可各选权重，结束后自动保存棋谱。" : mode !== "ai" ? "同屏轮流操作，绿色先手，金色后手。" :
        engine === "alphazero" ? "载入权重后开始新对局。" : "设置将在新对局生效。";
    updateWatchOptions();
}

function updateWatchOptions() {
    for (let side = 0; side < 2; side++)
        $("watch-model-field-" + side).hidden = $("watch-algorithm-" + side).value !== "alphazero";
}

function watchPlayers() {
    return [0, 1].map(side => {
        const path = $("watch-model-" + side).value;
        const structure = path.match(/b(\d+)-c(\d+)/i);
        return {algorithm: $("watch-algorithm-" + side).value, difficulty: $("watch-difficulty-" + side).value,
            model_path: path, residual_blocks: structure ? Number(structure[1]) : Number($("blocks").value),
            channels: structure ? Number(structure[2]) : Number($("channels").value)};
    });
}

function selectWatchSettings(players, cap = 400) {
    if (!players) return;
    players.forEach((agent, side) => {
        $("watch-algorithm-" + side).value = agent.algorithm;
        $("watch-difficulty-" + side).value = agent.difficulty;
        if (agent.model) $("watch-model-" + side).value = agent.model.name;
    });
    $("watch-max").value = cap;
    updateWatchOptions();
}

function stopWatch() {
    watching = false;
    clearTimeout(watchTimer);
    updateControls();
    updateStatus();
}

function queueWatch() {
    clearTimeout(watchTimer);
    if (watching && !busy && !replayData && state?.mode === "watch" && !state.finished)
        watchTimer = setTimeout(watchStep, Number($("watch-speed").value));
}

async function watchStep() {
    if (busy || replayData || state?.mode !== "watch" || state.finished) return;
    setBusy(true, `${state.current_player === 0 ? "绿色" : "金色"} ${playerName(state.current_player)} 正在思考…`);
    try {
        render(await request("/api/ai", {id: state.id}));
    } catch (error) {
        stopWatch();
        toast(error.message, true);
    } finally {
        setBusy(false);
    }
}

function selectAISettings(engine, level, simulations = null) {
    $("algorithm").value = engine || "alphazero";
    $("difficulty").value = level || "normal";
    if (simulations !== null) {
        const selector = $("simulations");
        if (![...selector.options].some(option => Number(option.value) === simulations))
            selector.add(new Option(`自定义 · ${simulations} 次搜索`, String(simulations)));
        selector.value = String(simulations);
    }
    updateAIOptions();
}

function selectHuman(value) {
    humanPlayer = value;
    ["first", "second"].forEach((id, index) => {
        $(id).classList.toggle("selected", index === value);
        $(id).setAttribute("aria-pressed", String(index === value));
    });
}

function selectTool(value) {
    tool = value;
    ["move", "horizontal", "vertical"].forEach(entry => {
        $("tool-" + entry).classList.toggle("selected", entry === value);
        $("tool-" + entry).setAttribute("aria-pressed", String(entry === value));
    });
    drawBoard();
    updateStatus();
}

async function playAI() {
    if (!state || state.finished || state.winner !== null || state.mode !== "ai" || state.human_turn || replayData) return;
    setBusy(true, "AI 正在思考…");
    render(await request("/api/ai", {
        id: state.id
    }));
}

async function startMatch(useCurrent = false) {
    if (busy) return;
    stopWatch();
    stopReplay();
    setBusy(true, "正在准备新对局…");
    try {
        const chosen = useCurrent && state && !replayData ? {
            mode: state.mode,
            human_player: state.human_player,
            simulations: state.simulations,
            algorithm: state.algorithm,
            difficulty: state.difficulty,
            max_moves: state.max_moves,
            players: state.players?.map(agent => ({...agent, model_path: agent.model?.name,
                residual_blocks: agent.model?.residual_blocks, channels: agent.model?.channels}))
        } : {
            mode,
            human_player: humanPlayer,
            simulations: Number($("simulations").value),
            algorithm: $("algorithm").value,
            difficulty: $("difficulty").value,
            players: watchPlayers(), max_moves: Number($("watch-max").value)
        };
        const next = await request("/api/new", chosen);
        replayData = null;
        liveState = null;
        render(next);
        selectMode(chosen.mode);
        selectHuman(chosen.human_player);
        if (state.mode === "watch") selectWatchSettings(state.players, state.max_moves);
        else selectAISettings(state.algorithm, state.difficulty, state.simulations);
        selectTool("move");
        if (state.mode === "watch") watching = true;
        else await playAI();
    } catch (error) {
        toast(error.message, true);
    } finally {
        setBusy(false);
    }
}

async function submitAction(action) {
    if (busy || !state || state.finished || state.winner !== null || !state.human_turn || replayData) return;
    if (!state.legal_actions.includes(action)) {
        toast(action < 81 ? "请选择高亮的合法落点。" : "这里不能放墙，请选择绿色预览的位置。", true);
        return;
    }
    if ((tool === "move") !== (action < 81)) return;
    setBusy(true, "正在落子…");
    try {
        render(await request("/api/move", {
            id: state.id,
            action
        }));
        await playAI();
    } catch (error) {
        toast(error.message, true);
    } finally {
        setBusy(false);
    }
}

function guessStructure(name) {
    const match = name.match(/b(\d+)-c(\d+)/i);
    if (match) {
        $("blocks").value = match[1];
        $("channels").value = match[2];
    }
}

function modelLabel(path) {
    return (path.includes("/archive/") ? "历史 · " : "") + path.split("/").pop().replace(/^[0-9a-f]{12}-/i, "");
}

function showModel(model) {
    $("model-state").classList.toggle("loaded", !!model);
    $("model-name").textContent = model ?
        `${modelLabel(model.name)} · ${model.residual_blocks} 块 / ${model.channels} 通道` : "尚未载入模型";
}

async function refreshModels(initial = false) {
    const response = await request("/api/status"),
        selector = $("model-select"),
        previous = selector.value;
    selector.replaceChildren();
    if (!response.models.length) selector.add(new Option("暂无模型，可从电脑选择权重", ""));
    else response.models.forEach(name => selector.add(new Option(modelLabel(name), name)));
    if (response.models.includes(previous)) selector.value = previous;
    else if (response.model && response.models.includes(response.model.name)) selector.value = response.model.name;
    showModel(response.model);
    for (let side = 0; side < 2; side++) {
        const choices = $("watch-model-" + side), previousPath = choices.value;
        choices.replaceChildren(new Option(response.model ? "使用已载入模型" : "请选择模型", ""));
        response.models.forEach(path => choices.add(new Option(modelLabel(path), path)));
        choices.value = response.models.includes(previousPath) ? previousPath : response.models[0] || "";
    }
    algorithms = response.algorithms || algorithms;
    difficulties = response.difficulties || difficulties;
    if (initial) {
        $("blocks").value = response.defaults.residual_blocks;
        $("channels").value = response.defaults.channels;
        selectAISettings(response.defaults.algorithm, response.defaults.difficulty);
    }
    guessStructure(selector.value);
    $("load").disabled = busy || !selector.value;
}

async function loadSelectedModel() {
    if (busy || !$("model-select").value) return;
    setBusy(true, "正在载入模型…");
    try {
        const response = await request("/api/model/load", {
            path: $("model-select").value,
            residual_blocks: Number($("blocks").value),
            channels: Number($("channels").value)
        });
        showModel(response.model);
        selectAISettings("alphazero", $("difficulty").value);
        toast("模型已载入，已选择 AlphaZero，可开始新对局。");
    } catch (error) {
        toast(error.message, true);
    } finally {
        setBusy(false);
    }
}

async function uploadModel(file) {
    if (!file || busy) return;
    if (!file.name.toLowerCase().endsWith(".h5")) {
        toast("请选择 .h5 权重文件。", true);
        return;
    }
    if (file.size > 128 * 1024 * 1024) {
        toast("模型文件不能超过 128 MiB。", true);
        return;
    }
    guessStructure(file.name);
    setBusy(true, "正在上传并载入模型…");
    try {
        const query = new URLSearchParams({
            name: file.name,
            residual_blocks: $("blocks").value,
            channels: $("channels").value
        });
        const response = await fetch(`/api/model/upload?${query}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/octet-stream"
            },
            body: file
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "模型上传失败。");
        await refreshModels();
        $("model-select").value = result.model.name;
        showModel(result.model);
        selectAISettings("alphazero", $("difficulty").value);
        toast("权重已载入，已选择 AlphaZero，可开始新对局。");
    } catch (error) {
        toast(error.message, true);
    } finally {
        $("model-file").value = "";
        setBusy(false);
    }
}

async function refreshRecords() {
    const response = await request("/api/records");
    const selector = $("saved-records"), previous = selector.value;
    selector.replaceChildren();
    if (!response.records.length) selector.add(new Option("暂无已保存的棋谱", ""));
    response.records.forEach(record => {
        const date = record.saved_at ? new Date(record.saved_at).toLocaleString("zh-CN") : "";
        const result = record.winner === 0 ? "绿色胜" : record.winner === 1 ? "金色胜" :
            record.end_reason === "move_limit" ? "平局" : "未结束";
        selector.add(new Option(`${date} · ${record.moves} 步 · ${result}`, record.name));
    });
    if (response.records.some(record => record.name === previous)) selector.value = previous;
    updateControls();
}

function downloadRecord(record, name) {
    const url = URL.createObjectURL(new Blob([JSON.stringify(record, null, 2)], {type: "application/json;charset=utf-8"}));
    const link = document.createElement("a");
    link.href = url;
    link.download = name;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function saveRecord() {
    if (busy || !state) return;
    if (replayData) {
        downloadRecord(replayData.record, "quoridor-replay.json");
        return;
    }
    setBusy(true, "正在保存棋谱…");
    try {
        const result = await request("/api/record/save", {id: state.id});
        render(result.state);
        downloadRecord(result.record, `quoridor-${result.filename}`);
        await refreshRecords();
        toast("棋谱已保存到 records/，并开始下载。");
    } catch (error) {
        toast(error.message, true);
    } finally {
        setBusy(false);
    }
}

function stopReplay() {
    replayPlaying = false;
    clearTimeout(replayTimer);
    updateControls();
}

function showReplay(index) {
    if (!replayData) return;
    const record = replayData.record, length = record.moves.length;
    replayIndex = Math.max(0, Math.min(length, index));
    const frame = replayData.frames[replayIndex];
    $("replay-slider").max = length;
    $("replay-slider").value = replayIndex;
    $("replay-position").textContent = `${replayIndex} / ${length} 步`;
    render({...frame, id: null, mode: record.mode, human_player: record.human_player,
        players: record.players, moves: record.moves.slice(0, replayIndex), move_count: replayIndex,
        human_turn: false, can_undo: false,
        ai_stats: [...record.moves.slice(0, replayIndex)].reverse().find(move => move.search)?.search || null,
        max_moves: record.max_moves,
        finished: replayIndex === length && record.end_reason !== "unfinished", end_reason: record.end_reason});
    const result = record.winner === 0 ? "绿色获胜" : record.winner === 1 ? "金色获胜" :
        record.end_reason === "move_limit" ? "达到步数上限，平局" : "对局未结束";
    $("record-status").textContent = `棋谱结果：${result}。返回对局后可继续游戏。`;
}

function enterReplay(data) {
    stopWatch();
    stopReplay();
    if (!replayData) liveState = state;
    replayData = data;
    showReplay(0);
}

function queueReplay() {
    clearTimeout(replayTimer);
    if (!replayPlaying || !replayData) return;
    if (replayIndex >= replayData.record.moves.length) {
        stopReplay();
        return;
    }
    replayTimer = setTimeout(() => {
        showReplay(replayIndex + 1);
        queueReplay();
    }, 600);
}

async function importRecord(file) {
    if (!file || busy) return;
    stopWatch();
    if (file.size > 2 * 1024 * 1024) {
        toast("棋谱文件不能超过 2 MiB。", true);
        return;
    }
    setBusy(true, "正在校验棋谱…");
    try {
        const record = JSON.parse(await file.text());
        enterReplay(await request("/api/record/import", {record}));
        toast("棋谱已载入，可按步查看或自动播放。");
    } catch (error) {
        toast(`载入棋谱失败：${error.message}`, true);
    } finally {
        $("record-file").value = "";
        setBusy(false);
    }
}

async function exitReplay() {
    if (busy || !replayData) return;
    stopReplay();
    replayData = null;
    render(liveState);
    setBusy(true, "正在返回对局…");
    try {
        render(await request(`/api/state?id=${encodeURIComponent(liveState.id)}`));
    } catch (error) {
        toast(error.message, true);
    } finally {
        liveState = null;
        setBusy(false);
    }
}

$("mode-ai").onclick = () => selectMode("ai");
$("mode-pvp").onclick = () => selectMode("pvp");
$("mode-watch").onclick = () => selectMode("watch");
for (let side = 0; side < 2; side++) $("watch-algorithm-" + side).onchange = updateWatchOptions;
$("watch-toggle").onclick = () => {
    if (watching) stopWatch();
    else {
        watching = true;
        updateControls();
        updateStatus();
        queueWatch();
    }
};
$("watch-step").onclick = () => { stopWatch(); watchStep(); };
$("watch-speed").onchange = queueWatch;
$("save-record").onclick = saveRecord;
$("import-record").onclick = () => { stopWatch(); $("record-file").click(); };
$("record-file").onchange = event => importRecord(event.target.files[0]);
$("records-panel").ontoggle = () => {
    if ($("records-panel").open) refreshRecords().catch(error => toast(error.message, true));
};
$("saved-records").onchange = updateControls;
$("open-saved-record").onclick = async () => {
    if (busy || !$("saved-records").value) return;
    stopWatch();
    setBusy(true, "正在打开棋谱…");
    try {
        enterReplay(await request(`/api/record?name=${encodeURIComponent($("saved-records").value)}`));
    } catch (error) {
        toast(error.message, true);
    } finally {
        setBusy(false);
    }
};
for (const [id, position] of [["replay-first", () => 0], ["replay-prev", () => replayIndex - 1],
    ["replay-next", () => replayIndex + 1], ["replay-last", () => replayData.record.moves.length]]) {
    $(id).onclick = () => { stopReplay(); showReplay(position()); };
}
$("replay-slider").oninput = () => { stopReplay(); showReplay(Number($("replay-slider").value)); };
$("replay-play").onclick = () => {
    if (replayPlaying) stopReplay();
    else {
        if (replayIndex >= replayData.record.moves.length) showReplay(0);
        replayPlaying = true;
        updateControls();
        queueReplay();
    }
};
$("replay-exit").onclick = exitReplay;
$("first").onclick = () => selectHuman(0);
$("second").onclick = () => selectHuman(1);
$("algorithm").onchange = updateAIOptions;
$("difficulty").onchange = () => {
    $("simulations").value = String({easy: 64, normal: 128, hard: 256, expert: 512}[$("difficulty").value]);
    updateAIOptions();
};
["move", "horizontal", "vertical"].forEach(value => {
    $("tool-" + value).onclick = () => selectTool(value);
});
$("start").onclick = () => startMatch();
$("restart").onclick = () => startMatch(true);
$("load").onclick = loadSelectedModel;
$("model-select").onchange = () => {
    guessStructure($("model-select").value);
    $("load").disabled = !$("model-select").value;
};
$("refresh-models").onclick = async () => {
    if (busy) return;
    try {
        await refreshModels();
        toast("模型列表已刷新。");
    } catch (error) {
        toast(error.message, true);
    }
};
$("upload").onclick = () => $("model-file").click();
$("model-file").onchange = event => uploadModel(event.target.files[0]);
$("undo").onclick = async () => {
    if (busy || !state?.can_undo) return;
    stopWatch();
    setBusy(true, "正在悔棋…");
    try {
        render(await request("/api/undo", {
            id: state.id
        }));
    } catch (error) {
        toast(error.message, true);
    } finally {
        setBusy(false);
    }
};
$("retry-ai").onclick = async () => {
    if (busy) return;
    try {
        await playAI();
    } catch (error) {
        toast(error.message, true);
    } finally {
        setBusy(false);
    }
};
$("board").addEventListener("click", event => {
    const target = event.target.closest("[data-action]");
    if (target) submitAction(Number(target.getAttribute("data-action")));
});
$("board").addEventListener("keydown", event => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const target = event.target.closest("[data-action]");
    if (target) {
        event.preventDefault();
        submitAction(Number(target.getAttribute("data-action")));
    }
});

async function initialize() {
    setBusy(true, "正在准备棋盘…");
    try {
        await Promise.all([refreshModels(true), refreshRecords()]);
        const id = sessionStorage.getItem("quoridor-match");
        if (id) {
            try {
                render(await request(`/api/state?id=${encodeURIComponent(id)}`));
            } catch {
                sessionStorage.removeItem("quoridor-match");
            }
        }
        if (!state) render(await request("/api/new", {
            mode: "pvp",
            human_player: 0,
            simulations: 128
        }));
        selectMode(state.mode);
        selectHuman(state.human_player);
        if (state.mode === "watch") selectWatchSettings(state.players, state.max_moves);
        else selectAISettings(state.algorithm, state.difficulty, state.simulations);
        await playAI();
        $("service-text").textContent = "本地服务已连接";
    } catch (error) {
        $("service-text").textContent = "本地服务未连接";
        $("service-dot").style.background = "#bd6660";
        toast(error.message, true);
    } finally {
        setBusy(false);
    }
}
initialize();
