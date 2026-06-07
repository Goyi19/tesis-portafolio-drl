```mermaid
graph TD
    %% Estilos
    classDef agent fill:#1E293B,stroke:#38BDF8,stroke-width:2px,color:#fff;
    classDef env fill:#0F172A,stroke:#10B981,stroke-width:2px,color:#fff;
    classDef data fill:#334155,stroke:#94A3B8,stroke-width:1px,color:#fff;
    classDef action fill:#FB923C,stroke:#EA580C,stroke-width:2px,color:#fff;
    classDef state fill:#8B5CF6,stroke:#7C3AED,stroke-width:2px,color:#fff;
    classDef reward fill:#F43F5E,stroke:#E11D48,stroke-width:2px,color:#fff;

    subgraph Agente_DRL ["Cerebro DRL (PPO / SAC)"]
        Actor["Actor (Política πθ)<br/>Input: o_t<br/>Output: Gaussiana (μ, σ)"]
        Critic["Crítico (Valor)<br/>PPO: V(s_t)<br/>SAC: min(Q1, Q2)"]
        Update["Optimizador (ADAM)<br/>PPO: Clipping (20%) + Ventaja<br/>SAC: Max Entropía (αH)"]
        
        Actor -.-> Update
        Critic -.-> Update
    end
    class Agente_DRL,Actor,Critic,Update agent

    subgraph Entorno_Mercado ["PortfolioEnv (Mercado)"]
        Transicion["Dinámica de Capital:<br/>x_{t+1} = x_t * (1 + Σ w_i R_i - costos)"]
        CalcReward["Cálculo de Recompensa (λ=0.10):<br/>r_t = Σ w_i μ_{mix,i} - λ ΣΣ w_i w_j σ_{mix,ij} - costos"]
        DataHandler["DataHandler<br/>(Carga features de t+1)"]
        
        Transicion --> CalcReward
        CalcReward --> DataHandler
    end
    class Entorno_Mercado,Transicion,CalcReward,DataHandler env

    %% Conexiones principales del loop RL
    Actor -- "Acción (a_t)<br/>Pesos Proyectados<br/>w_i = max(0, a_i) / Σ max(0, a_j)" --> Transicion
    
    DataHandler -- "Estado Siguiente (o_{t+1})<br/>[x_{t+1}, w_t, t/T, norm_μ, norm_σ]" --> Actor
    DataHandler -.-> Critic
    
    CalcReward -- "Recompensa (r_t)<br/>Señal de media-varianza" --> Critic

    %% Estilos de enlaces
    linkStyle 0,1 stroke:#38BDF8,stroke-width:2px;
    linkStyle 2,3 stroke:#10B981,stroke-width:2px;
    linkStyle 4 stroke:#FB923C,stroke-width:3px,color:#FB923C;
    linkStyle 5,6 stroke:#8B5CF6,stroke-width:3px,color:#8B5CF6;
    linkStyle 7 stroke:#F43F5E,stroke-width:3px,color:#F43F5E;
```
