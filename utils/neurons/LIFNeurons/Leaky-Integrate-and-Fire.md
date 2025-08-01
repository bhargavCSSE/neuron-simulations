# 🧠 Leaky Integrate-and-Fire (LIF) Neuron Model

This document describes the implementation and behavior of the **LIFNeuron** class used in the simulator pipeline.

---

## 🔍 Overview

The **Leaky Integrate-and-Fire (LIF)** neuron is one of the simplest and most commonly used spiking neuron models. It captures basic neuronal behavior such as:

- Integrating input current over time
- Leaky decay of the membrane potential
- Emitting a spike when a threshold is crossed
- Resetting after a spike

---

## 📐 Biological Inspiration

In biological neurons:

- The **membrane potential** builds up as the neuron receives input.
- If the potential crosses a certain **threshold**, the neuron "fires" (spikes).
- After firing, the membrane potential is **reset**, and the neuron continues integrating.

This model approximates that behavior using the following simplified equation:

## 🔄 LIF Neuron Simulation Flow

```mermaid
flowchart TD
    A[Start of Step] --> B[Receive Input Current I(t)]
    B --> C[Update Membrane Potential v(t)]

    C --> D{v(t) ≥ Threshold?}
    D -- Yes --> E[Emit Spike: spike = 1]
    E --> F[Reset v(t) to v_reset]
    F --> G[Clear Input Current]
    G --> H[End of Step]

    D -- No --> I[No Spike: spike = 0]
    I --> G
```

## 🧮 LIF Neuron Equation

The membrane potential `v(t)` evolves over time using the following update rule:

$$
v(t + 1) = v(t) + \frac{1}{\tau} \left( - \left( v(t) - v_{\text{rest}} \right) + I(t) \right)
$$

Where:

- \( v(t) \): Membrane potential at time step \( t \)
- \( \tau \): Membrane time constant (leak rate)
- \( v_{\text{rest}} \): Resting membrane potential
- \( I(t) \): Input current at time step \( t \)

### Spike Condition:

If \( v(t + 1) \geq v_{\text{thresh}} \), the neuron emits a spike and resets:

$$
\text{if } v(t+1) \geq v_{\text{thresh}} \Rightarrow \text{spike} = 1, \quad v(t+1) = v_{\text{reset}}
$$
