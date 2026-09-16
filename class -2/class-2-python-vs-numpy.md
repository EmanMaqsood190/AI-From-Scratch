<div align="center">

[![Typing SVG](https://readme-typing-svg.demolab.com/?font=Fira+Code&size=24&pause=800&color=00E0B0&center=true&vCenter=true&multiline=false&width=850&lines=%F0%9F%90%8D+Pure+Python+vs+%F0%9F%93%A6+NumPy;Same+math.+Same+answer.+Different+effort.;Spoiler%3A+the+library+didn't+do+the+thinking+%F0%9F%98%8E)](https://git.io/typing-svg)

![Round](https://img.shields.io/badge/ROUND-1-black?style=for-the-badge)
![Pure Python](https://img.shields.io/badge/🐍_Pure_Python-loss_=_9-3776AB?style=for-the-badge)
![VS](https://img.shields.io/badge/VS-⚔️-red?style=for-the-badge)
![NumPy](https://img.shields.io/badge/📦_NumPy-loss_=_9-013243?style=for-the-badge)

</div>

---

<div align="center">

```text
        🐍                          📦
   PURE PYTHON                    NUMPY
   "I do it myself"          "I do it in bulk"
        |                            |
        +------------ ⚔️ ------------+
                      |
              SAME EXACT MATH
```

</div>

---

# 🥊 The Setup

Same numbers for both fighters. No tricks.

<div align="center">

| `x` | `weight` | `bias` | `target` |
|:---:|:---:|:---:|:---:|
| 2 | 3 | 1 | 10 |

</div>

```mermaid
flowchart LR
    X["x = 2"] --> M["× weight (3)"] --> B["+ bias (1)"] --> P["prediction = 7"]
    P --> E["error = 7 − 10 = −3"] --> L["loss = (−3)² = 9"]
    style P fill:#1f4d7d,color:#fff
    style L fill:#7d1f1f,color:#fff
```

---

# 🐍 Corner 1 — Without a Library

<details open>
<summary><b>▶️ pure_python.py</b></summary>

```python
# Input
x = 2

# Weight and bias
weight = 3
bias = 1

# Correct answer
target = 10

# Prediction
prediction = weight * x + bias

# Error
error = prediction - target

# Loss
loss = error ** 2

print("Prediction:", prediction)
print("Error:", error)
print("Loss:", loss)
```

</details>

```text
Prediction: 7
Error: -3
Loss: 9
```

---

# 📦 Corner 2 — With NumPy

<details open>
<summary><b>▶️ with_numpy.py</b></summary>

```python
import numpy as np

# Input
x = 2

# Weight and bias
weight = 3
bias = 1

# Correct answer
target = 10

# Prediction
prediction = weight * x + bias

# Error
error = prediction - target

# Loss
loss = np.square(error)

print("Prediction:", prediction)
print("Error:", error)
print("Loss:", loss)
```

</details>

```text
Prediction: 7
Error: -3
Loss: 9
```

---

# 🤔 So What's Actually Different?

<div align="center">

### Almost nothing. 😭

</div>

| Step | 🐍 Pure Python | 📦 NumPy | Same? |
|---|---|---|:---:|
| Prediction | `weight * x + bias` | `weight * x + bias` | ✅ |
| Error | `prediction - target` | `prediction - target` | ✅ |
| **Loss** | `error ** 2` | `np.square(error)` | ⚠️ *different spelling* |
| Result | `9` | `9` | ✅ |

<div align="center">

```text
      error ** 2                np.square(error)
           \                          /
            \                        /
             +--------→  (-3)² = 9  ←+
                     identical
```

</div>

> 🧠 One line changed. The math didn't.

---

# 🚀 Where NumPy Actually Earns Its Keep

The moment you stop having **one** number and start having **many**:

```python
x = [2, 4, 6]
weights = [3, 1, 2]
```

<table>
<tr>
<th>🐍 Pure Python — manual labor</th>
<th>📦 NumPy — one line</th>
</tr>
<tr>
<td>

```python
result = []
for i in range(len(x)):
    result.append(x[i] * weights[i])

print(result)
```

</td>
<td>

```python
import numpy as np

x = np.array([2, 4, 6])
weights = np.array([3, 1, 2])

result = x * weights

print(result)
```

</td>
</tr>
<tr>
<td align="center"><code>[6, 4, 12]</code></td>
<td align="center"><code>[ 6  4 12]</code></td>
</tr>
</table>

```mermaid
flowchart LR
    subgraph "🐍 one at a time"
    a1["2 × 3"] --> a2["4 × 1"] --> a3["6 × 2"]
    end
    subgraph "📦 all at once"
    b1["[2, 4, 6] × [3, 1, 2]"]
    end
    a3 --> R["[6, 4, 12]"]
    b1 --> R
    style R fill:#1f7d3a,color:#fff
```

<div align="center">

```text
3 numbers      → loops are fine        🙂
300 numbers    → loops are annoying    😐
3,000,000      → loops are 💀          NumPy 📦
```

</div>

---

<div align="center">

# 🏆 The Verdict

</div>

> ### **Libraries don't change the math.**
> ### They give us tools to perform the math more easily — especially when the amount of data gets bigger.

<div align="center">

```text
   🐍 shows you WHAT is happening
   📦 saves you time WHILE it happens
   
   Learn with 🐍.  Ship with 📦.
```

![Verdict](https://img.shields.io/badge/Winner-Both_(it's_the_same_math)-success?style=for-the-badge)

</div>

---

<div align="center">

### 🎯 One-Line Recap

`error ** 2` and `np.square(error)` are twins — NumPy just scales to a million of them without breaking a sweat.

</div>
