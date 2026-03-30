\chapter{基于Colocation Benefit Score的动态调度算法设计}

第三章完成了MLWD的结构设计、离线采集实现以及干扰系数规则化估算方法的构建，为刻画推理任务的GPU资源消耗模式并估算共置干扰提供了基础能力。本章在此基础上，提出动态调度算法CBS（Colocation Benefit Score），实现在线调度决策。CBS的核心思想是：对于每个待调度的推理请求，量化将其与已有任务共置相对于路由至专用节点（即严格PD分离）的系统级净收益，当CBS得分为正时选择共置路径，否则选择分离路径。

本章首先建立PD动态调度的形式化问题模型与优化目标，然后分别推导分离路径与共置路径的成本模型，并在此基础上给出CBS评分公式与决策规则，进一步描述CBS的完整调度流程，最后通过退化性分析证明CBS在极端负载条件下能够自动退化为纯分离或纯共置策略。

\section{问题建模与优化目标}

本节建立PD动态调度的形式化问题模型。与将Prefill和Decode静态绑定到固定节点不同，本文允许调度器在每个请求到达时动态选择执行路径：分离模式（Strict Disaggregation）或共置模式（Co-location）。

\subsection{PD调度的决策空间}

\textbf{集群模型。}考虑一个由 $N$ 个GPU节点组成的LLM推理服务集群 $\mathcal{N} = \{n_1, n_2, \dots, n_N\}$，每个节点 $n_i$ 包含一块或多块同构GPU。集群采用PD分离架构作为基础部署模式，在任意时刻 $t$，节点被划分为Prefill节点集合 $\mathcal{P}(t)$ 以及Decode节点集合 $\mathcal{D}(t)$，满足 $\mathcal{P}(t) \cup \mathcal{D}(t) = \mathcal{N}$ 且 $\mathcal{P}(t) \cap \mathcal{D}(t) = \emptyset$。节点角色在初始化时静态配置，但可通过4.5.6节描述的角色自适应机制在运行时动态调整。节点间通过高速网络互联，支持KV Cache跨节点传输。

\textbf{请求模型。}请求流 $\mathcal{R} = \{r_1, r_2, \dots\}$ 是一个按到达时间排序的推理请求序列，每个请求 $r_j$ 由以下属性刻画：到达时间 $a_j$；输入序列长度 $s_j$；预期输出长度 $\hat{o}_j$（通过历史分布估计）；所请求的模型标识 $m_j$ 及其对应的推理框架 $\text{fw}_j$。

\textbf{节点状态模型。}在任意时间 $t$，每个节点 $n_i$ 的状态由以下变量决定：当前正在执行的请求集合 $\mathcal{W}_i(t)$，其中每个任务包含其MLWD向量 $\mathbf{W}_w$；显存剩余率 $\rho_{\mathrm{mem},i}(t)$；排队队列长度 $Q_i(t)$；当前Decode批处理大小 $B_i(t)$。

\textbf{决策空间。}对于每个到达的请求 $r_j$，调度器面临两层决策：

第一层是执行模式选择：$\operatorname{mode}(r_j) \in \{\text{DISAGGREGATE},\ \text{COLOCATE}\}$。

分离模式（DISAGGREGATE）：请求 $r_j$ 的Prefill阶段在某个Prefill节点 $p \in \mathcal{P}$ 上独立执行，完成后将KV Cache通过网络传输至某个Decode节点 $d \in \mathcal{D}$，Decode阶段在 $d$ 上执行。Prefill和Decode分别在专用资源上运行，互不干扰，但会引入Prefill排队等待、KV Cache传输以及控制面协调等额外开销。

共置模式（COLOCATE）：请求 $r_j$ 的Prefill阶段被调度至某个正在执行Decode任务的节点 $d \in \mathcal{D}$，通过GPU共享机制与已有Decode任务共享GPU资源。Prefill无需等待专用Prefill节点空闲，也无需传输KV Cache，但Prefill与Decode之间存在资源竞争导致的相互干扰。当推理框架启用Chunked Prefill时，共置模式的决策空间进一步扩展：调度器不仅决定是否共置，还需决定以多大的chunk大小 $C$ 共置。chunk大小 $C$ 作为部署配置上下文中的 $C_{\text{chunk}}$ 参数，在构建MLWD时决定了算子运行时画像层的实际取值——不同的 $C$ 值对应不同的Kernel执行模式和资源占用画像，因此第三章的加权映射规则能够反映 $\hat{\alpha}_p$ 和 $\hat{\alpha}_d$ 随 $C$ 变化的关系——当 $C$ 足够小时，Prefill chunk可以填充Decode阶段的空闲计算单元，估算得到的 $\hat{\alpha}_d$ 趋近于零。调度器通过对若干候选 $C$ 值分别构建MLWD并代入加权映射规则，选择使CBS最大化的 $C$ 值。

第二层是节点选择：在确定模式后，选择具体的目标节点。本文将节点选择与模式选择解耦——模式选择由CBS算法决定，节点选择在同一模式内采用加权负载均衡策略。综合负载定义为：
\begin{equation}
\operatorname{Load}(n_i, t) = \beta_1 \cdot \frac{|\mathcal{W}_i(t)|}{B_{\mathrm{max}}} + \beta_2 \cdot (1 - \rho_{\mathrm{mem},i}(t)) + \beta_3 \cdot \frac{Q_i(t)}{Q_{\mathrm{max}}}
\end{equation}

其中 $\beta_1, \beta_2, \beta_3$ 为权重系数（本文取 $\beta_1=0.4, \beta_2=0.4, \beta_3=0.2$，通过网格搜索在验证集上确定），$B_{\mathrm{max}}$ 和 $Q_{\mathrm{max}}$ 分别为最大批处理容量和最大队列长度。该公式仅用于同一模式内的节点排序，不参与模式选择决策。


\subsection{优化目标：Goodput最大化与延迟约束}

Goodput定义为单位时间窗口内成功完成且满足SLO约束的请求数量：
\begin{equation}
\operatorname{Goodput}(T) = \frac{1}{T} \left| \left\{ r_j \in \mathcal{R}_T : \operatorname{TTFT}(r_j) \leq \mathrm{SLO}_{\mathrm{TTFT}} \wedge \operatorname{TPOT}(r_j) \leq \mathrm{SLO}_{\mathrm{TPOT}} \right\} \right|
\end{equation}

其中 $\mathcal{R}_T$ 表示在时间窗口 $T$ 内完成的所有请求集合。TTFT和TPOT分别定义为：
\begin{equation}
\operatorname{TTFT}(r_j) = t_j^{\mathrm{first\_token}} - a_j
\end{equation}
\begin{equation}
\operatorname{TPOT}(r_j) = \frac{t_j^{\mathrm{last\_token}} - t_j^{\mathrm{first\_token}}}{s_j^{\mathrm{output}} - 1}
\end{equation}

本文采用P99延迟作为SLO约束：
\begin{equation}
P_{99}(\operatorname{TTFT}) \leq \mathrm{SLO}_{\mathrm{TTFT}}, \quad P_{99}(\operatorname{TPOT}) \leq \mathrm{SLO}_{\mathrm{TPOT}}
\end{equation}

PD动态调度对TTFT和TPOT的影响路径不同。在分离模式下：
\begin{equation}
\operatorname{TTFT}_{\mathrm{disagg}}(r_j) = T_{\mathrm{queue}}^{(p)}(r_j) + T_{\mathrm{prefill}}^{(0)}(r_j) + T_{\mathrm{kv}}(r_j)
\end{equation}
\begin{equation}
\operatorname{TPOT}_{\mathrm{disagg}}(r_j) = T_{\mathrm{decode\_step}}^{(0)}(r_j)
\end{equation}

在共置模式下：
\begin{equation}
\operatorname{TTFT}_{\mathrm{coloc}}(r_j, d) = T_{\mathrm{queue}}^{(d)}(r_j) + T_{\mathrm{prefill}}^{(0)}(r_j) \cdot (1 + \hat{\alpha}_p(r_j, d))
\end{equation}
\begin{equation}
\operatorname{TPOT}_{\mathrm{coloc}}(r_j, d) = T_{\mathrm{decode\_step}}^{(0)}(r_j) \cdot (1 + \hat{\alpha}_d(d, r_j))
\end{equation}

其中 $T_{\mathrm{prefill}}^{(0)}$ 和 $T_{\mathrm{decode\_step}}^{(0)}$ 分别为无干扰基线Prefill时延和单步Decode时延（由离线Profiling查找表或参数化算子模型获取），$T_{\mathrm{kv}}$ 为KV Cache跨节点传输时延。$\hat{\alpha}_p(r_j, d)$ 和 $\hat{\alpha}_d(d, r_j)$ 分别为第三章 3.4.3 节定义的节点级 Prefill 干扰系数和节点级 Decode 干扰系数。由于不同的chunk大小 $C$ 对应不同的算子运行时画像（MLWD第一层），加权映射规则的结果隐含了 $C$ 的影响——不同的 $C$ 值对应不同的MLWD特征取值，从而产生不同的 $\hat{\alpha}_p$ 和 $\hat{\alpha}_d$ 估算值。

综上，本文的优化问题形式化为：
\begin{equation}
\begin{aligned}
&\max_{\{\operatorname{mode}(r_j)\}_{r_j \in \mathcal{R}}} \operatorname{Goodput}(T) \\
\text{s.t.} \quad & P_{99}(\operatorname{TTFT}) \leq \mathrm{SLO}_{\mathrm{TTFT}} \\
& P_{99}(\operatorname{TPOT}) \leq \mathrm{SLO}_{\mathrm{TPOT}} \\
& \forall n_i \in \mathcal{N},\ \forall t:\ \rho_{\mathrm{mem},i}(t) \geq \rho_{\mathrm{min}}
\end{aligned}
\end{equation}

最后一个约束为显存安全约束（$\rho_{\mathrm{min}} = 0.05$），确保共置操作不会导致OOM。

此外，本文引入 compute budget 约束，限制单个 iteration 中处理的 token 总量。在批处理 Decode 中，一个 iteration 同时为 $|\mathcal{W}_d^{\mathrm{dec}}|$ 个请求各生成一个 token，iteration 时延即为各请求的 TPOT。当 Chunked Prefill 向同一 iteration 注入额外 token 时，iteration 时延近似按 token 总量线性增长。据此，由 TPOT SLO 反推每个 iteration 最多可处理的 token 数：
\begin{equation}
B_{\mathrm{token}}^{\max}(d, t) = \left\lfloor \frac{\mathrm{SLO}_{\mathrm{TPOT}}}{\bar{T}_{\mathrm{iter}}^{(0)}(d, t)\, /\, |\mathcal{W}_d^{\mathrm{dec}}(t)|} \right\rfloor
\end{equation}

其中 $\bar{T}_{\mathrm{iter}}^{(0)}(d, t)$ 为节点 $d$ 在当前 Decode 批大小下的无干扰 iteration 基线时延（可从离线查找表获取），$\bar{T}_{\mathrm{iter}}^{(0)}(d, t) / |\mathcal{W}_d^{\mathrm{dec}}(t)|$ 为每个 token 的边际处理时间。即使显存充足且 CBS 为正，如果当前 iteration 的 token 总量（decode tokens + prefill chunk tokens）已超过 $B_{\mathrm{token}}^{\max}$，也应拒绝共置。该约束确保共置决策不会因为过度填充单个 iteration 而导致所有请求的 TPOT 恶化。

该优化问题是一个在线随机优化问题：请求到达时间和输出长度均不可预知，调度决策需在请求到达后短时间内完成。CBS算法将全局优化问题分解为逐请求的贪心决策，通过量化每个请求在共置模式下相对于分离模式的即时净收益，在每个决策点做出局部最优选择。

\section{分离路径与共置路径的成本建模}

CBS评分的核心是比较两条路径的成本差异。本节统一定义分离路径和共置路径的成本模型，为4.3节的CBS公式推导提供基础。

\subsection{分离路径成本}

分离路径的额外成本相对于"理想零开销执行"的增量，主要由三部分构成：
\begin{equation}
C_{\mathrm{disagg}}(r_j) = T_{\mathrm{queue}}^{(p)}(r_j) + T_{\mathrm{kv}}(r_j) + T_{\mathrm{ctrl}}
\end{equation}

其中：
$T_{\mathrm{queue}}^{(p)}(r_j)$ 为Prefill节点上的排队等待时延，可通过在线监控系统实时获取当前队列长度和平均服务时间估计：
\begin{equation}
T_{\mathrm{queue}}^{(p)}(r_j) \approx Q_p(a_j) \cdot \bar{T}_{\mathrm{prefill}}^{(p)}
\end{equation}
$T_{\mathrm{kv}}(r_j)$ 为KV Cache跨节点传输时延：
\begin{equation}
T_{\mathrm{kv}}(r_j) = \frac{\operatorname{Size}_{\mathrm{kv}}(r_j)}{\mathrm{BW}_{\mathrm{eff}}} + T_{\mathrm{setup}}
\end{equation}

其中 $\operatorname{Size}_{\mathrm{kv}}(r_j) = 2 \cdot L \cdot n_{\mathrm{kv}} \cdot d_{\mathrm{head}} \cdot s_j \cdot \operatorname{sizeof}(\mathrm{dtype})$，可由部署配置上下文中的模型架构参数和请求的序列长度 $s_j$ 解析计算；$\mathrm{BW}_{\mathrm{eff}}$ 为有效网络带宽；$T_{\mathrm{setup}}$ 为传输初始化开销。

$T_{\mathrm{ctrl}}$ 为控制面协调开销（Prefill-Decode握手、KV Cache地址注册等），通常为常数级别（毫秒量级）。

\subsection{共置路径成本}

共置路径的成本体现为资源竞争带来的时延增长，包含两个分量：新请求自身的Prefill时延膨胀，以及对节点上已有Decode请求的外部性代价。以下公式中的干扰系数采用第三章 3.4.3 节定义的两种记法：$\hat{\alpha}_p(r_j, d)$ 为节点级 Prefill 干扰系数，由节点 $d$ 上已有 Decode 负载的聚合 MLWD 计算得到；$\hat{\alpha}_d(u, r_j)$ 为逐对 Decode 干扰系数，保留各请求的个体差异以支持 SLO 紧迫度加权。

\textbf{Prefill时延膨胀。}新请求 $r_j$ 在节点 $d$ 上共置执行时，其Prefill时延相较无干扰基线的增长量为：
\begin{equation}
\Delta T_{\mathrm{prefill}}(r_j, d) = \hat{\alpha}_p(r_j, d) \cdot T_{\mathrm{prefill}}^{(0)}(r_j)
\end{equation}

\textbf{Decode外部性代价。}设节点 $d$ 在时刻 $a_j$ 上已有Decode请求集合为 $\mathcal{W}_d^{\mathrm{dec}}(a_j) = \{u_1, u_2, \dots, u_M\}$。新请求的Prefill共置执行期间，这些已有请求的单步Decode时延膨胀为 $T_{\mathrm{decode\_step}}^{(0)}(u) \cdot (1 + \hat{\alpha}_d(u, r_j))$。

在Prefill持续的时间区间 $\tau_j^{(d)} \approx T_{\mathrm{prefill}}^{(0)}(r_j) \cdot (1 + \hat{\alpha}_p(r_j, d))$ 内，每个已有请求 $u$ 受到的额外代价近似为：
\begin{equation}
\Delta_{\mathrm{ext}}(u \mid r_j, d) \approx \tau_j^{(d)} \cdot \hat{\alpha}_d(u, r_j)
\end{equation}

节点上所有已有Decode请求的总外部性代价为：
\begin{equation}
\Delta_{\mathrm{ext}}(r_j, d) = \sum_{u \in \mathcal{W}_d^{\mathrm{dec}}(a_j)} \omega_u \cdot \tau_j^{(d)} \cdot \hat{\alpha}_d(u, r_j)
\end{equation}

其中权重 $\omega_u$ 反映不同请求的SLO紧迫度：
\begin{equation}
\omega_u = \eta_1 + \eta_2 \cdot \frac{\hat{o}_u^{\mathrm{remain}}}{\bar{o}} + \eta_3 \cdot \frac{\mathrm{TPOT}_u}{\mathrm{SLO}_{\mathrm{TPOT}}}
\end{equation}

即剩余输出越长、当前TPOT越接近SLO的请求，其被拖慢的代价越大。

综合上述分量，共置路径的总成本为：
\begin{equation}
C_{\mathrm{coloc}}(r_j, d) = T_{\mathrm{queue}}^{(d)}(r_j) + \Delta T_{\mathrm{prefill}}(r_j, d) + \lambda \cdot \Delta_{\mathrm{ext}}(r_j, d) + \Delta_{\mathrm{dispatch}}(r_j, d)
\end{equation}

其中 $T_{\mathrm{queue}}^{(d)}(r_j)$ 为候选 Decode 节点 $d$ 上的排队等待时延。共置模式下，新请求的 Prefill 需要等待被调度进目标节点的当前 iteration，因此同样存在排队开销，该项与 4.1.2 节中 $\operatorname{TTFT}_{\mathrm{coloc}}$ 的定义一致。$\lambda > 0$ 为外部性权重，控制系统对已有Decode请求服务质量的保护程度。$\Delta_{\mathrm{dispatch}}(r_j, d)$ 为dispatch争抢延迟项，建模共置时CPU-GPU dispatch通道的额外争抢开销。BestServe的研究指出，Decode阶段的瓶颈往往不是显存带宽而是CPU-GPU dispatch latency，当两个任务共置时，Kernel Launch的频率翻倍，dispatch争抢可能成为新的瓶颈。该项定义为：
\begin{equation}
\Delta_{\mathrm{dispatch}}(r_j, d) = \kappa \cdot \frac{1}{\bar{g}_{\mathrm{launch}}^{(r_j)}} \cdot \frac{1}{\bar{g}_{\mathrm{launch}}^{(d)}} \cdot \tau_j^{(d)}
\end{equation}

其中 $\kappa$ 为dispatch争抢系数（量纲为 $\mu\text{s}^2$，使 $\Delta_{\mathrm{dispatch}}$ 的结果量纲为时延），$\bar{g}_{\mathrm{launch}}^{(r_j)}$ 和 $\bar{g}_{\mathrm{launch}}^{(d)}$ 分别为新请求和节点已有任务的Kernel Launch间隔（来自MLWD第一层，单位为 $\mu\text{s}$），$\tau_j^{(d)}$ 为共置 Prefill 的持续时间（定义见式(4.5)）。当两个任务的Kernel Launch间隔都很小时（如vLLM的细粒度Kernel调度），dispatch争抢项显著增大；当某个任务使用TensorRT-LLM等算子融合框架时（$\bar{g}_{\mathrm{launch}}$ 较大），该项较小。


\section{CBS评分公式与决策规则}

\subsection{CBS评分公式}

基于4.2节的成本模型，CBS评分定义为分离路径成本与共置路径成本之差，同时引入约束风险惩罚项：
\begin{equation}
\operatorname{CBS}(r_j, d) = \begin{cases} -\infty, & \mathbb{I}_{\mathrm{mem}}(r_j, d) = 0 \\ C_{\mathrm{disagg}}(r_j) - C_{\mathrm{coloc}}(r_j, d) - \mu \cdot \Delta_{\mathrm{risk}}(r_j, d), & \text{otherwise} \end{cases}
\end{equation}

其中 $\mathbb{I}_{\mathrm{mem}}$ 为显存可行性判定函数：
\begin{equation}
\mathbb{I}_{\mathrm{mem}}(r_j, d) = \begin{cases} 1, & \rho_{\mathrm{mem},d}(a_j) - \hat{m}(r_j) \geq \rho_{\mathrm{min}} \\ 0, & \text{otherwise} \end{cases}
\end{equation}

$\hat{m}(r_j)$ 为新请求Prefill共置时需要占用的额外显存比例，由部署配置上下文中的模型架构参数和请求序列长度解析计算。

$\Delta_{\mathrm{risk}}$ 为SLO风险惩罚项：
\begin{equation}
\begin{aligned}
\Delta_{\mathrm{risk}}(r_j,d)
&= \gamma_1 \cdot \max\!\left(
0,\,
\widehat{\operatorname{TTFT}}_{\mathrm{coloc}}(r_j,d) - \mathrm{SLO}_{\mathrm{TTFT}}
\right)
\\
&\quad + \gamma_2 \cdot
\sum_{u \in \mathcal{W}_d^{\mathrm{dec}}(a_j)}
\max\!\left(
0,\,
\widehat{\operatorname{TPOT}}(u \mid r_j,d) - \mathrm{SLO}_{\mathrm{TPOT}}
\right)
\end{aligned}
\end{equation}

其中 $\widehat{\operatorname{TTFT}}_{\mathrm{coloc}}$ 和 $\widehat{\operatorname{TPOT}}$ 是基于MLWD干扰模型的在线预测值。该项确保即便平均收益为正，只要存在显著的SLO违约风险，CBS也会被拉低。$\mu > 0$ 为风险惩罚权重。

将成本模型代入并展开，CBS的工程计算形式为：

\begin{equation}
\begin{aligned}
\operatorname{CBS}(r_j, d) \approx\;&
\underbrace{
\left(T_{\mathrm{queue}}^{(p)} - T_{\mathrm{queue}}^{(d)}\right) + T_{\mathrm{kv}}(r_j) + T_{\mathrm{ctrl}}
}_{\text{排队节省与KV本地化收益}}
-
\underbrace{
\hat{\alpha}_p(r_j, d) \cdot T_{\mathrm{prefill}}^{(0)}(r_j)
}_{\text{Prefill干扰损失}} \\
&-
\underbrace{
\lambda \sum_{u \in \mathcal{W}_d^{\mathrm{dec}}}
\omega_u \cdot \tau_j^{(d)} \cdot \hat{\alpha}_d(u, r_j)
}_{\text{Decode外部性代价}}
-
\underbrace{
\Delta_{\mathrm{dispatch}}(r_j, d)
}_{\text{Dispatch争抢损失}}
-
\underbrace{
\mu \cdot \Delta_{\mathrm{risk}}(r_j, d)
}_{\text{SLO风险惩罚}}
\end{aligned}
\end{equation}

其中，$T_{\mathrm{queue}}^{(p)} - T_{\mathrm{queue}}^{(d)}$ 来自 $C_{\mathrm{disagg}}$ 中的 Prefill 排队时延与 $C_{\mathrm{coloc}}$ 中的 Decode 排队时延之差，反映共置路径在排队环节的相对节省；$T_{\mathrm{kv}}(r_j) + T_{\mathrm{ctrl}}$ 为分离路径独有的 KV Cache 传输与控制面协调开销，共置路径无需承担。

由上式可知，CBS 量化了共置相对于分离的净收益。正向收益来自三项：共置路径省下的排队时延差 $T_{\mathrm{queue}}^{(p)} - T_{\mathrm{queue}}^{(d)}$、免去的 KV Cache 传输时延 $T_{\mathrm{kv}}$ 以及控制面协调开销 $T_{\mathrm{ctrl}}$。负向代价包括 Prefill 自身被拖慢、对同节点 Decode 请求的群体拖慢、dispatch 争抢损失以及 SLO 违约风险惩罚。其中 $\hat{\alpha}_p(r_j, d)$ 和 $\hat{\alpha}_d(u, r_j)$ 均由第三章的 MLWD 驱动的加权映射规则给出，不同的 chunk 大小 $C$ 对应不同的 MLWD 算子画像（第一层），因此估算结果隐含了 $C$ 的影响。当推理框架启用 Chunked Prefill 时，调度器对若干候选 $C$ 值分别构建 MLWD 并代入加权映射规则，选择使 CBS 最大化的 $C$ 值。

\subsection{CBS公式各分量的在线获取方式}

CBS公式中的各分量均通过可观测或可计算的方式在线获取：$T_{\mathrm{queue}}^{(p)}$ 和 $T_{\mathrm{queue}}^{(d)}$ 由在线监控子系统实时采集各节点的队列长度和平均服务时间，并通过 Prometheus 指标管线上报。$T_{\mathrm{kv}}(r_j)$ 由部署配置上下文中的模型架构参数以及请求序列长度 $s_j$ 解析计算，并结合网络带宽监控值进行估计。$T_{\mathrm{prefill}}^{(0)}(r_j)$ 和 $T_{\mathrm{decode\_step}}^{(0)}(u)$ 由离线 Profiling 构建的基线延迟查找表获取，其中以 $(b, s)$ 作为索引。$\hat{\alpha}_p$ 和 $\hat{\alpha}_d$ 由第三章设计的 MLWD 驱动的加权映射规则计算，仅涉及向量内积运算，在线开销可忽略；不同的chunk大小 $C$ 对应不同的MLWD算子画像（第一层），因此估算结果隐含了 $C$ 的影响。$\rho_{\mathrm{mem},d}$、$Q_d$ 以及 $|\mathcal{W}_d^{\mathrm{dec}}|$ 则由节点侧在线监控 Agent 周期性采集并上报。

\subsection{多候选节点下的决策规则}

由于一个请求通常对应多个可行的Decode候选节点，模式选择与节点选择统一为如下两步：

第一步，在所有满足显存安全约束的候选Decode节点集合上计算CBS分数：
\begin{equation}
\mathcal{D}_j^{\mathrm{feasible}} = \{d \in \mathcal{D} \mid \mathbb{I}_{\mathrm{mem}}(r_j, d) = 1\}
\end{equation}
\begin{equation}
d_j^{\star} = \arg\max_{d \in \mathcal{D}_j^{\mathrm{feasible}}} \operatorname{CBS}(r_j, d)
\end{equation}

第二步，执行模式判定：
\begin{equation}
\operatorname{mode}(r_j) = \begin{cases} \text{COLOCATE on } d_j^{\star}, & \max_{d \in \mathcal{D}_j^{\mathrm{feasible}}} \operatorname{CBS}(r_j, d) > 0 \\ \text{DISAGGREGATE}, & \text{otherwise} \end{cases}
\end{equation}

当选择分离模式时，在Prefill节点集合 $\mathcal{P}$ 中按 $\operatorname{Load}(n_i, t)$ 选择负载最低的节点。

该规则将"是否共置"与"共置到哪个节点"统一为一个最大化问题：只有当存在某个候选Decode节点使得共置净收益为正时，系统才执行共置；否则自动退回严格PD分离。


\section{CBS动态调度流程}

CBS动态调度流程包括候选节点过滤、路径评分和节点选择三个阶段。算法\ref{alg:cbs}给出了完整的伪代码描述。

\RestyleAlgo{ruled}
\begin{algorithm}[htbp]
\caption{CBS动态调度算法}
\label{alg:cbs}
\KwData{请求 $r_j$，Decode候选节点集合 $\mathcal{D}$，Prefill节点集合 $\mathcal{P}$，静态MLWD库 $\mathcal{L}$，在线状态集合 $\mathcal{S}$}
\KwResult{请求 $r_j$ 的目标节点与调度路径}

构建请求 $r_j$ 的MLWD向量 $\mathbf{W}_{r_j}$\;
$g^\star \leftarrow -\infty,\ d^\star \leftarrow \mathrm{null}$\;

\ForEach{$d \in \mathcal{D}$}{
    \If{$\mathbb{I}_{\mathrm{mem}}(r_j,d)=0$}{continue\;}
    \If{当前 iteration token 总量 $> B_{\mathrm{token}}^{\max}(d, t)$}{continue\;}

    读取节点 $d$ 的聚合MLWD $\bar{\mathbf{W}}_d$\;
    $(\hat{\alpha}_p,\hat{\alpha}_d) \leftarrow \mathrm{EstimateIF}(\mathbf{W}_{r_j},\bar{\mathbf{W}}_d, \mathbf{w})$\
    计算 $C_{\mathrm{disagg}}(r_j)$ 与 $C_{\mathrm{coloc}}(r_j,d)$\;
    $g \leftarrow C_{\mathrm{disagg}}(r_j)-C_{\mathrm{coloc}}(r_j,d)-\mu\cdot\Delta_{\mathrm{risk}}(r_j,d)$\;

    \If{$g > g^\star$}{
        $g^\star \leftarrow g,\ d^\star \leftarrow d$\;
    }
}

\If{$g^\star > 0$}{
    \Return{$(d^\star,\ \text{共置路径})$}\;
}
$n_p^\star \leftarrow \arg\min_{n\in\mathcal{P}} \mathrm{Load}(n,t)$\;
\Return{$(n_p^\star,\ \text{分离路径})$}\;
\end{algorithm}

从伪代码可以看出，CBS本质上是一种以分离为基础、低干扰共置为投机补充的动态策略。它并不盲目追求更高的共置比例，而是在每个请求到达时依据即时成本做出局部最优选择，从而在系统吞吐与服务质量之间取得平衡。

\section{基于SLO违反概率与分层阈值的双向迁移纠错机制}

CBS 属于一次性贪心决策机制，即在请求到达时决定采用共置还是分离，后续不再调整。然而，系统运行过程中仍可能出现两类难以在调度时准确预见的情况：一类是干扰强于预期，或突发流量导致节点过载，此时需要将请求从高负载节点迁出以缓解风险；另一类是部分 Decode 节点随着请求完成而逐渐空闲，资源利用率下降，此时需要将请求合并到其他节点，以释放节点资源。为提高系统鲁棒性并改善资源利用率，本文在 CBS 之外进一步引入基于 SLO 违反概率与分层阈值的双向迁移机制，使调度过程具备在线纠错和资源整合能力。

\subsection{SLO违反概率的定义与计算}

对于节点 $d$ 上执行的 Decode 请求 $u$，定义其 SLO 违反概率为
\begin{equation}
\begin{aligned}
P_{\mathrm{viol}}(u,d,t)
&= P\Bigl(
\operatorname{TPOT}(u) > \mathrm{SLO}_{\mathrm{TPOT}}
\,\Bigm|\,
\mathcal{W}_d(t)
\Bigr)
\end{aligned}
\end{equation}

该概率基于 MLWD 驱动的规则化估算方法计算。具体做法是，根据节点上除 $u$ 以外的共置任务的聚合 MLWD 向量，按第三章 3.4.3 节的节点级公式计算干扰系数 $\hat{\alpha}_d(u, d, t)$（此时聚合范围为 $\mathcal{W}_d(t) \setminus \{u\}$），再结合无干扰条件下的基线时延 $T_{\mathrm{decode\_step}}^{(0)}(u)$，估计共置条件下的 TPOT：
\begin{equation}
\widehat{\operatorname{TPOT}}(u, d, t) = T_{\mathrm{decode\_step}}^{(0)}(u) \cdot (1 + \hat{\alpha}_d(u, d, t))
\end{equation}

考虑到规则化估算存在误差，设实际干扰系数服从以 $\hat{\alpha}_d$ 为均值、以历史估算残差标准差 $\sigma_{\alpha}$ 为标准差的正态分布，则 SLO 违反概率可近似写为
\begin{equation}
P_{\mathrm{viol}}(u, d, t) \approx 1 - \Phi\left(\frac{\mathrm{SLO}_{\mathrm{TPOT}} / T_{\mathrm{decode\_step}}^{(0)}(u) - 1 - \hat{\alpha}_d}{\sigma_{\alpha}}\right)
\end{equation}

其中，$\Phi$ 表示标准正态分布的累积分布函数。

\subsection{迁移开销的在线测量与建模}

在线迁移的核心操作是将请求 $u$ 的 KV Cache 从源节点 $d$ 迁移到目标节点 $d'$，并在迁移完成后切换执行位置。该过程会引入额外开销，若忽略这一部分代价，迁移可能反而降低系统整体性能。因此，迁移决策需要同时考虑迁移收益和迁移成本。

迁移开销可分为三个部分。

第一部分为 KV Cache 传输时延 $T_{\mathrm{mig}}^{\mathrm{kv}}$。请求 $u$ 的 KV Cache 大小由模型参数和当前序列长度 $s_u(t)$ 决定：
\begin{equation}
\operatorname{Size}_{\mathrm{kv}}(u, t) = 2 \cdot L \cdot n_{\mathrm{kv}} \cdot d_{\mathrm{head}} \cdot s_u(t) \cdot \operatorname{sizeof}(\mathrm{dtype})
\end{equation}

相应的传输时延为
\begin{equation}
T_{\mathrm{mig}}^{\mathrm{kv}}(u, t) = \frac{\operatorname{Size}_{\mathrm{kv}}(u, t)}{\mathrm{BW}_{\mathrm{avail}}(d, d', t)} + T_{\mathrm{setup}}^{\mathrm{mig}}
\end{equation}

其中，$\mathrm{BW}_{\mathrm{avail}}(d, d', t)$ 表示时刻 $t$ 下源节点与目标节点之间的可用带宽，$T_{\mathrm{setup}}^{\mathrm{mig}}$ 表示迁移初始化开销，包括显存分配、地址注册和上下文准备等操作。与静态传输模型不同，迁移发生在系统运行过程中，链路带宽可能同时受到其他传输任务影响，因此可用带宽需要通过在线监测估计。

第二部分为源节点带宽争用代价 $\Delta_{\mathrm{bw}}^{\mathrm{src}}$。迁移期间，KV Cache 传输会占用源节点出口带宽，进而影响该节点上其他 KV 传输任务。其代价定义为迁移导致的额外传输时延增量：
\begin{equation}
\Delta_{\mathrm{bw}}^{\mathrm{src}}(d, t) = \sum_{v \in \mathcal{T}_d^{\mathrm{kv}}(t)} \left(\frac{\operatorname{Size}_{\mathrm{kv}}(v)}{\mathrm{BW}_{\mathrm{avail}}(d, \cdot, t) - \mathrm{BW}_{\mathrm{mig}}} - \frac{\operatorname{Size}_{\mathrm{kv}}(v)}{\mathrm{BW}_{\mathrm{avail}}(d, \cdot, t)}\right)
\end{equation}

其中，$\mathcal{T}_d^{\mathrm{kv}}(t)$ 为源节点上正在进行的 KV 传输任务集合，$\mathrm{BW}_{\mathrm{mig}}$ 为迁移占用带宽。若源节点不存在其他并发传输，则该项为零。

第三部分为目标节点接纳代价 $\Delta_{\mathrm{admit}}^{\mathrm{dst}}$。请求迁入目标节点后，会增加该节点 Decode 批处理规模，从而可能导致已有请求的 TPOT 上升。该代价通过干扰系数估算公式计算：
\begin{equation}
\Delta_{\mathrm{admit}}^{\mathrm{dst}}(d', u, t) = \sum_{v \in \mathcal{W}_{d'}^{\mathrm{dec}}(t)} T_{\mathrm{decode\_step}}^{(0)}(v) \cdot \left[\hat{\alpha}_d(v, d' \cup {u}, t) - \hat{\alpha}_d(v, d', t)\right]
\end{equation}

据此，迁移总开销定义为
\begin{equation}
C_{\mathrm{mig}}(u, d, d', t) = T_{\mathrm{mig}}^{\mathrm{kv}}(u, t) + \xi_1 \cdot \Delta_{\mathrm{bw}}^{\mathrm{src}}(d, t) + \xi_2 \cdot \Delta_{\mathrm{admit}}^{\mathrm{dst}}(d', u, t)
\end{equation}

其中，$\xi_1$ 和 $\xi_2$ 为权重系数，用于平衡各部分开销的重要性。上述模型中的可用带宽、迁移初始化开销和并发传输状态均依赖在线监测结果，从而使迁移决策建立在实时系统状态之上，而不是静态假设之上。

\subsection{迁移净收益}

迁移决策需要同时考虑迁移带来的性能改善和迁移本身的开销。为使两者可比，本文将迁移收益也定义在时延量纲上。

将请求 $u$ 从源节点 $d$ 迁移到目标节点 $d'$ 后，源节点上剩余请求的 TPOT 改善量为：
\begin{equation}
\Delta_{\mathrm{src}}^{+}(u, d, t) = \sum_{v \in \mathcal{W}_d^{\mathrm{dec}}(t) \setminus \{u\}} T_{\mathrm{decode\_step}}^{(0)}(v) \cdot \left[\hat{\alpha}_d(v, d, t) - \hat{\alpha}_d(v, d \setminus \{u\}, t)\right]
\end{equation}

即移除 $u$ 后，源节点上各请求因干扰减轻而获得的单步 Decode 时延缩短之和。

据此，迁移净收益定义为源节点改善量减去迁移总开销：
\begin{equation}
G_{\mathrm{mig}}(u, d, d', t) = \hat{o}_{\mathrm{remain}}^{(u)} \cdot \Delta_{\mathrm{src}}^{+}(u, d, t) - C_{\mathrm{mig}}(u, d, d', t)
\end{equation}

其中，$\hat{o}_{\mathrm{remain}}^{(u)}$ 为请求 $u$ 的预估剩余输出 token 数，将单步改善量放大到请求剩余生命周期，使收益项与 $C_{\mathrm{mig}}$ 在时延量纲上可比。$G_{\mathrm{mig}} > 0$ 表示迁移的预期收益超过开销，系统应执行该迁移。

\subsection{基于分层阈值的双向迁移触发}

迁移控制模块（Rescheduler）以固定周期（本文取1秒）扫描所有Decode节点，根据4.5.5节定义的分层阈值体系触发两个方向的迁移：

\textbf{Mitigation方向。}当节点 $d$ 上存在请求 $u$ 满足 $P_{\mathrm{viol}}(u, d, t) > \theta_{\mathrm{ceil}}$ 时，触发迁出迁移。对每个候选目标节点 $d'$ 计算 4.5.3 节定义的迁移净收益 $G_{\mathrm{mig}}(u, d, d', t)$，选择 $G_{\mathrm{mig}}$ 最大且为正的节点作为迁移目标。

\textbf{Consolidation方向。}当节点 $d$ 上所有请求的预测TPOT均满足 $\widehat{\operatorname{TPOT}}(u, d, t) < \theta_{\mathrm{floor}} \cdot \mathrm{SLO}_{\mathrm{TPOT}}$ 时，触发合并迁移。目标节点采用bin-packing策略，在满足接纳后预测TPOT不超过 $\theta_{\mathrm{dispatch}} \cdot \mathrm{SLO}_{\mathrm{TPOT}}$ 的安全节点中，选择接纳后预测TPOT最高的节点，使低负载节点尽快清空。

Mitigation优先于consolidation执行。在迁移期间，请求的Decode继续在源节点执行，KV Cache异步传输至目标节点，传输完成后在下一个iteration切换执行节点。各阈值的具体取值与含义见4.5.5节，迁移并发控制见4.5.7节，节点角色自适应见4.5.6节。

\subsection{基于分层阈值的双向迁移策略}

迁移触发条件采用三层阈值体系。

第一层为迁出上限阈值 $\theta_{\mathrm{ceil}}$。当节点 $d$ 上存在请求 $u$ 满足 $P_{\mathrm{viol}}(u, d, t) > \theta_{\mathrm{ceil}}$ 时，触发缓解型迁移，将该请求迁移到负载较低的节点。该阈值用于识别过载风险，本文取 $\theta_{\mathrm{ceil}} = 0.3$。

第二层为合并下限阈值 $\theta_{\mathrm{floor}}$。当节点 $d$ 上所有请求的预测 TPOT 均满足 $\widehat{\operatorname{TPOT}}(u, d, t) < \theta_{\mathrm{floor}} \cdot \mathrm{SLO}_{\mathrm{TPOT}}$ 时，将该节点视为低负载节点，并触发整合型迁移，将其请求合并到其他 Decode 节点。本文取 $\theta_{\mathrm{floor}} = 0.4$。

第三层为调度安全阈值 $\theta_{\mathrm{dispatch}}$。对整合型迁移而言，目标节点在接纳新请求后必须满足预测 TPOT 不超过 $\theta_{\mathrm{dispatch}} \cdot \mathrm{SLO}_{\mathrm{TPOT}}$，以为后续波动预留余量。本文取 $\theta_{\mathrm{dispatch}} = 0.85$。

三层阈值满足 $\theta_{\mathrm{floor}} < \theta_{\mathrm{dispatch}} < \theta_{\mathrm{ceil}}$。其中，$\theta_{\mathrm{floor}}$ 和 $\theta_{\mathrm{dispatch}}$ 直接以 TPOT 占 SLO 的比例表示，而 $\theta_{\mathrm{ceil}}$ 通过 SLO 违反概率定义，在同一量纲下可对应更高的 TPOT/SLO 水平。通过这种分层设置，可以区分过载缓解与资源整合两类不同迁移动机。

\subsection{基于Consolidation的节点角色自适应}

静态的 Prefill:Decode 节点划分存在天然局限。当 Prefill 请求增加时，Prefill 节点可能成为瓶颈，而部分 Decode 节点仍处于低负载状态；反之亦然。仅依赖逐请求共置决策，难以从全局上调整两类节点的资源比例。

为此，可利用 consolidation 迁移后被清空的 Decode 节点，执行轻量级角色转换。

当某个 Decode 节点 $d$ 上的请求全部被合并到其他节点后，若 Prefill 节点的平均排队时延 $\bar{T}_{\mathrm{queue}}^{(\mathcal{P})}(t)$ 超过阈值 $T_{\mathrm{role}}^{\mathrm{thresh}}$，则将 $d$ 临时转换为 Prefill 节点：
$\text{if } \bar{T}_{\mathrm{queue}}^{(\mathcal{P})}(t) > T_{\mathrm{role}}^{\mathrm{thresh}} \text{ and } |\mathcal{W}_d(t)| = 0: \quad d \in \mathcal{D} \rightarrow d \in \mathcal{P}$

其中，$T_{\mathrm{role}}^{\mathrm{thresh}}$ 取为 $0.5 \cdot \mathrm{SLO}_{\mathrm{TTFT}}$。

相反，当临时转为 Prefill 的节点完成当前任务后，若 Decode 节点的平均预测 TPOT 超过 $\theta_{\mathrm{dispatch}} \cdot \mathrm{SLO}_{\mathrm{TPOT}}$，则将该节点回收为 Decode 节点：
$\text{if } \bar{\widehat{\operatorname{TPOT}}}^{(\mathcal{D})}(t) > \theta_{\mathrm{dispatch}} \cdot \mathrm{SLO}_{\mathrm{TPOT}} \text{ and } |\mathcal{W}_d^{\mathrm{prefill}}(t)| = 0: \quad d \in \mathcal{P} \rightarrow d \in \mathcal{D}$

为保证系统具备基本服务能力，还需设置保留节点约束，即始终满足 $|\mathcal{P}| \geq 1$ 且 $|\mathcal{D}| \geq 1$。在此条件下，节点角色可以随负载变化动态调整，从而改善 Prefill 与 Decode 之间的资源配比。

\subsection{迁移并发控制}

迁移会占用网络带宽和控制面资源。若在短时间内触发过多迁移，可能引起迁移风暴，反而导致系统性能下降。因此，需要对迁移并发进行约束。

系统设置全局迁移速率上限 $R_{\max}$，即每个扫描周期内最多执行 $R_{\max}$ 次迁移。本文取 $R_{\max}=2$。其中，mitigation 与 consolidation 共用同一配额，但优先执行 mitigation。同时，设置 KV Cache 传输总量上限 $V_{\max}^{\mathrm{kv}}$，要求同时进行的迁移满足
$\sum_{(u, d \rightarrow d') \in \mathcal{M}_{\mathrm{active}}} \operatorname{Size}_{\mathrm{kv}}(u, t) \leq V_{\max}^{\mathrm{kv}}$,
本文取
$V_{\max}^{\mathrm{kv}} = 0.5 \cdot \mathrm{BW}_{\mathrm{link}} \cdot T_{\mathrm{scan}}$
以限制迁移对链路带宽的占用。

迁移完成后，设置请求级冷却时间 $T_{\mathrm{cool}}$。同一请求完成迁移后，在 $T_{\mathrm{cool}}$ 时间内不再参与新的迁移。本文取 $T_{\mathrm{cool}}=10$ s，以避免请求在节点间频繁往返。

当多个请求同时满足迁移条件时，还需要进一步确定执行顺序。对于 mitigation，按 $P_{\mathrm{viol}}$ 降序排序，优先处理 SLO 违反风险较高的请求；若多个请求的风险接近，则优先迁移 KV Cache 较小的请求，以降低迁移开销。对于 consolidation，则优先选择活跃请求数较少的低负载节点，以便更快完成节点清空和资源回收。

\subsection{双向迁移算法}

算法 \ref{alg:bilateral-migration} 给出了基于分层阈值的双向迁移流程。该算法周期性执行，与 CBS 初始调度相互配合：前者负责请求到达时的路径选择，后者负责在运行过程中根据系统状态对既有决策进行修正和整合。

\RestyleAlgo{ruled}
\begin{algorithm}[htbp]
\caption{基于分层阈值的双向迁移算法}
\label{alg:bilateral-migration}
\KwData{Decode节点集合 $\mathcal{D}$，Prefill节点集合 $\mathcal{P}$，迁出上限 $\theta_{\mathrm{ceil}}$，合并下限 $\theta_{\mathrm{floor}}$，调度安全阈值 $\theta_{\mathrm{dispatch}}$，迁移速率上限 $R_{\max}$，KV传输总量上限 $V_{\max}^{\mathrm{kv}}$，冷却期 $T_{\mathrm{cool}}$，角色转换阈值 $T_{\mathrm{role}}^{\mathrm{thresh}}$}
\KwResult{迁移决策集合 $\mathcal{M}_{\mathrm{mig}}$，角色转换决策集合 $\mathcal{R}_{\mathrm{role}}$}

$\mathcal{M}_{\mathrm{mig}} \leftarrow \emptyset,\ \mathcal{R}_{\mathrm{role}} \leftarrow \emptyset,\ V_{\mathrm{used}} \leftarrow 0$;

$\mathcal{C}_{\mathrm{mit}} \leftarrow \emptyset$;
\ForEach{$d \in \mathcal{D}$}{
\ForEach{$u \in \mathcal{W}_d^{\mathrm{dec}}(t)$ 且 $u$ 不在冷却期}{
计算 $P_{\mathrm{viol}}(u,d,t)$;
\If{$P_{\mathrm{viol}}(u,d,t) > \theta_{\mathrm{ceil}}$}{
$\mathcal{C}_{\mathrm{mit}} \leftarrow \mathcal{C}_{\mathrm{mit}} \cup {(u,d,P_{\mathrm{viol}},\mathrm{Size}_{\mathrm{kv}}(u,t))}$;
}
}
}

将 $\mathcal{C}_{\mathrm{mit}}$ 按 $P_{\mathrm{viol}}$ 降序排序
（若差值 $<0.05$，按 $\mathrm{Size}_{\mathrm{kv}}$ 升序）;

\ForEach{$(u,d,P_{\mathrm{viol}},\mathrm{sz}) \in \mathcal{C}_{\mathrm{mit}}$ 且 $|\mathcal{M}_{\mathrm{mig}}|<R_{\max}$ 且 $V_{\mathrm{used}}+\mathrm{sz}\leq V_{\max}^{\mathrm{kv}}$}{
对所有可行 $d' \in \mathcal{D}\setminus{d}$ 计算 $G_{\mathrm{mig}}(u,d,d',t)$;
$d^\star \leftarrow \arg\max G_{\mathrm{mig}}$;
\If{$d^\star \neq \mathrm{null}$ 且 $G_{\mathrm{mig}}(u,d,d^\star,t)>0$}{
$\mathcal{M}_{\mathrm{mig}} \leftarrow \mathcal{M}_{\mathrm{mig}} \cup {(u,d\rightarrow d^\star)}$;
$V_{\mathrm{used}} \leftarrow V_{\mathrm{used}} + \mathrm{sz}$;
标记 $u$ 冷却至 $T_{\mathrm{cool}}$;
}
}

$\mathcal{D}_{\mathrm{low}} \leftarrow {d \in \mathcal{D}\mid \forall u \in \mathcal{W}_d^{\mathrm{dec}}(t),\ \widehat{\mathrm{TPOT}}(u,d,t)<\theta_{\mathrm{floor}}\cdot \mathrm{SLO}_{\mathrm{TPOT}} \land |\mathcal{W}_d^{\mathrm{dec}}(t)|>0}$;
将 $\mathcal{D}_{\mathrm{low}}$ 按 $|\mathcal{W}_d^{\mathrm{dec}}(t)|$ 升序排序;

\ForEach{$d \in \mathcal{D}_{\mathrm{low}}$ 且 $|\mathcal{M}_{\mathrm{mig}}|<R_{\max}$ 且 $|\mathcal{D}|>1$}{
$\mathrm{all_placed} \leftarrow \mathrm{true}$;
\ForEach{$u \in \mathcal{W}_d^{\mathrm{dec}}(t)$ 且 $u$ 不在冷却期}{
$\mathcal{D}_{\mathrm{safe}} \leftarrow {d' \in \mathcal{D}\setminus{d}\mid \widehat{\mathrm{TPOT}}_{\mathrm{after}}(d',u,t)\leq \theta_{\mathrm{dispatch}}\cdot \mathrm{SLO}_{\mathrm{TPOT}} \land \text{显存充足}}$;
\If{$\mathcal{D}_{\mathrm{safe}}=\emptyset$ 或 $V_{\mathrm{used}}+\mathrm{Size}_{\mathrm{kv}}(u,t)>V_{\max}^{\mathrm{kv}}$}{
$\mathrm{all_placed} \leftarrow \mathrm{false}$;
break;
}
$d^\star \leftarrow \arg\max_{d' \in \mathcal{D}_{\mathrm{safe}}} \widehat{\mathrm{TPOT}}_{\mathrm{after}}(d',u,t)$;
$\mathcal{M}_{\mathrm{mig}} \leftarrow \mathcal{M}_{\mathrm{mig}} \cup {(u,d\rightarrow d^\star)}$;
$V_{\mathrm{used}} \leftarrow V_{\mathrm{used}} + \mathrm{Size}_{\mathrm{kv}}(u,t)$;
标记 $u$ 冷却至 $T_{\mathrm{cool}}$;
}
\If{$\mathrm{all_placed}$ 且 $\bar{T}_{\mathrm{queue}}^{(\mathcal{P})}(t)>T_{\mathrm{role}}^{\mathrm{thresh}}$ 且 $|\mathcal{D}|>1$}{
将 $d$ 从 $\mathcal{D}$ 移至 $\mathcal{P}$;
$\mathcal{R}_{\mathrm{role}} \leftarrow \mathcal{R}_{\mathrm{role}} \cup {d}$;
}
}

\Return{$\mathcal{M}_{\mathrm{mig}},\ \mathcal{R}_{\mathrm{role}}$};
\end{algorithm}
算法执行可分为三个阶段。第一阶段是 mitigation 迁移。系统扫描所有 Decode 节点上的活跃请求，筛选出 SLO 违反概率超过上限阈值 $\theta_{\mathrm{ceil}}$ 的请求，按优先级排序后依次选择收益最大的目标节点。第二阶段是 consolidation 迁移。系统识别预测 TPOT 全部低于阈值 $\theta_{\mathrm{floor}} \cdot \mathrm{SLO}_{\mathrm{TPOT}}$ 的低负载节点，并尝试将这些节点上的请求合并到其他满足安全阈值的节点上。第三阶段是节点角色自适应。当某个 Decode 节点被成功清空，且 Prefill 侧排队压力较高时，可将该节点暂时转为 Prefill 节点。

该算法的时间复杂度为 $O((|\mathcal{C}_{\mathrm{mit}}| + |\mathcal{D}_{\mathrm{low}}| \cdot \bar{B}) \cdot |\mathcal{D}|)$，其中 $\bar{B}$ 表示低负载节点的平均请求数。由于全局迁移速率和 KV 传输总量均受到约束，每个扫描周期内实际执行的迁移数目有限，因此整体计算开销是可控的。

\subsection{超参数汇总}

本章涉及的超参数按功能分为三组，汇总如表~\ref{tab:hyperparams} 所示。

\begin{table}[htbp]
\centering
\caption{CBS 动态调度算法超参数汇总}
\label{tab:hyperparams}
\small
\renewcommand{\arraystretch}{1.22}
\begin{tabularx}{\linewidth}{l c c X}
\toprule
超参数 & 符号 & 默认值 & 说明 \\
\midrule
\multicolumn{4}{l}{\textit{CBS 成本模型}} \\
外部性权重 & $\lambda$ & 1.0 & 控制 Decode 外部性代价在共置成本中的权重 \\
SLO 紧迫度基线 & $\eta_1$ & 0.5 & $\omega_u$ 中的常数项 \\
剩余长度权重 & $\eta_2$ & 0.3 & $\omega_u$ 中剩余输出长度的权重 \\
TPOT 紧迫度权重 & $\eta_3$ & 0.2 & $\omega_u$ 中当前 TPOT/SLO 比值的权重 \\
Dispatch 争抢系数 & $\kappa$ & 0.1 & Kernel Launch 争抢项的缩放系数 \\
风险惩罚权重 & $\mu$ & 2.0 & SLO 风险惩罚项 $\Delta_{\mathrm{risk}}$ 的权重 \\
TTFT 风险系数 & $\gamma_1$ & 1.0 & $\Delta_{\mathrm{risk}}$ 中 TTFT 违约部分的权重 \\
TPOT 风险系数 & $\gamma_2$ & 1.0 & $\Delta_{\mathrm{risk}}$ 中 TPOT 违约部分的权重 \\
\midrule
\multicolumn{4}{l}{\textit{双向迁移机制}} \\
迁出上限阈值 & $\theta_{\mathrm{ceil}}$ & 0.3 & SLO 违反概率超过此值触发 Mitigation \\
合并下限阈值 & $\theta_{\mathrm{floor}}$ & 0.4 & 预测 TPOT/SLO 低于此值触发 Consolidation \\
调度安全阈值 & $\theta_{\mathrm{dispatch}}$ & 0.85 & 目标节点接纳后 TPOT/SLO 不超过此值 \\
迁移带宽权重 & $\xi_1$ & 0.5 & $C_{\mathrm{mig}}$ 中源节点带宽争用的权重 \\
接纳代价权重 & $\xi_2$ & 1.0 & $C_{\mathrm{mig}}$ 中目标节点接纳代价的权重 \\
迁移速率上限 & $R_{\max}$ & 2 & 单个扫描周期内最大迁移数 \\
KV 传输总量上限 & $V_{\max}^{\mathrm{kv}}$ & $0.5 \cdot \mathrm{BW}_{\mathrm{link}} \cdot T_{\mathrm{scan}}$ & 单个扫描周期内 KV 传输总量上限 \\
冷却期 & $T_{\mathrm{cool}}$ & 10\,s & 请求迁移后的冷却时间 \\
角色转换阈值 & $T_{\mathrm{role}}^{\mathrm{thresh}}$ & $0.5 \cdot \mathrm{SLO}_{\mathrm{TTFT}}$ & Prefill 排队时延超过此值触发角色转换 \\
\bottomrule
\end{tabularx}
\end{table}

CBS 成本模型组的超参数（$\lambda$, $\eta_{1\text{-}3}$, $\kappa$, $\mu$, $\gamma_{1\text{-}2}$）通过在验证集上的网格搜索确定，优化目标为 goodput。双向迁移机制组的阈值参数（$\theta_{\mathrm{ceil}}$, $\theta_{\mathrm{floor}}$, $\theta_{\mathrm{dispatch}}$）基于 SLO 违反概率和 TPOT/SLO 比值的经验分布设定，并在第六章的敏感性分析中验证其鲁棒性。

\section{退化性分析}

本节证明CBS算法在极端负载条件下能够自动退化为纯分离或纯共置策略，从而说明CBS是对两类静态策略的统一泛化。

\textbf{退化为纯分离。}当所有Decode节点均处于高负载状态时，$\alpha_p$ 和 $\alpha_d$ 均较大，$\Delta_{\mathrm{ext}}$ 项显著增大，$\Delta_{\mathrm{dispatch}}$ 项因Kernel Launch频率翻倍而进一步增大，同时SLO风险项 $\Delta_{\mathrm{risk}}$ 也可能被触发。此时对所有候选节点 $d$，有 $\operatorname{CBS}(r_j, d) < 0$，调度器自动选择分离路径。在极端情况下，若所有Decode节点的显存均不满足约束（$\mathbb{I}_{\mathrm{mem}} = 0$）或token budget已耗尽，CBS直接返回 $-\infty$，系统完全退化为纯PD分离架构。

\textbf{退化为纯共置。}当Prefill节点严重拥塞（$T_{\mathrm{queue}}^{(p)} \gg 0$）且Decode节点较空闲（$\alpha_p, \alpha_d \approx 0$，$\Delta_{\mathrm{ext}} \approx 0$，$\Delta_{\mathrm{dispatch}} \approx 0$）时，CBS的排队节省收益和KV本地化收益远大于干扰损失，对所有可行Decode节点均有 $\operatorname{CBS}(r_j, d) > 0$，系统倾向于将所有新请求的Prefill共置到Decode节点上执行，行为等价于纯共置策略。

\textbf{统一性。}上述分析表明，纯分离和纯共置是CBS在参数空间两个极端区域的特例。CBS通过统一的净收益函数在不同负载区域内自适应切换，无需人工设定切换阈值或预先选择策略类型。

\section{本章小结}

本章提出了基于 Colocation Benefit Score（CBS）的 PD 动态调度算法。该算法通过比较分离路径与共置路径的即时代价，对每个到达请求实时选择 PD 分离或共置模式。在共置成本建模中，引入了 Chunked Prefill 感知、dispatch 争抢建模和 compute budget 约束，以实现更细致的共置代价估计。

在CBS基础决策之上，本章提出了基于SLO违反概率与分层阈值的双向迁移纠错机制。在 Mitigation 方向上，利用MLWD驱动的规则化估算方法实时计算各请求的SLO违反概率，当概率超过迁出上限阈值时，将高风险请求的 KV Cache 异步传输至低负载节点；在 Consolidation 方向上，当节点上所有请求的预测 TPOT 均低于合并下限阈值时，将请求合并到其他节点，并在 Prefill 排队压力较大时将被清空的节点临时转为 Prefill 节点。迁移并发通过全局速率限制、KV 传输总量限制和请求级冷却期进行约束。该机制使CBS从一次性贪心决策升级为具备在线纠错和资源整合能力的动态决策。

CBS的各分量均可通过在线可观测量或离线查找表获取，其中干扰系数 $\hat{\alpha}_p$ 和 $\hat{\alpha}_d$ 由第三章设计的MLWD驱动的加权映射规则计算。退化性分析表明，CBS在极端负载条件下自动退化为纯分离或纯共置策略，是对两类静态策略的统一泛化。

下一章将基于上述方法，在Kubernetes平台上给出Profiling-Scheduling系统的整体架构与关键模块实现。
