# 📚 Part 3 – Theory: Arithmetic of CNNs

## Question 1

**Given:** Input size = 32×32×3, filter size = 5×5, stride = 1, no padding, number of filters = 8  
**Output formula:**
\[
O = \left\lfloor \frac{(W - F + 2P)}{S} + 1 \right\rfloor
\]

- W = 32 (width)
- F = 5 (filter size)
- P = 0 (no padding)
- S = 1 (stride)

\[
O = \left\lfloor \frac{(32 - 5 + 0)}{1} + 1 \right\rfloor = \left\lfloor 28 \right\rfloor = 28
\]

So, output size is **28×28×8**

---

## Question 2

**Change:** Padding is now `"same"`  
With `"same"` padding and stride = 1, the output size remains the same as input:

So, output size = **32×32×8**

---

## Question 3

**Given:** Input = 64×64, filter = 3×3, stride = 2, no padding

\[
O = \left\lfloor \frac{(64 - 3)}{2} + 1 \right\rfloor = \left\lfloor \frac{61}{2} + 1 \right\rfloor = \left\lfloor 31.5 + 1 \right\rfloor = \left\lfloor 32.5 \right\rfloor = 32
\]

Output size = **32×32**

---

## Question 4

**Max Pooling:** 2×2, stride = 2, input = 16×16

\[
O = \left\lfloor \frac{(16 - 2)}{2} + 1 \right\rfloor = \left\lfloor \frac{14}{2} + 1 \right\rfloor = \left\lfloor 7 + 1 \right\rfloor = 8
\]

Output size = **8×8**

---

## Question 5

**Input:** 128×128  
**2 Conv Layers**: kernel = 3×3, stride = 1, padding = 'same'  

With `"same"` padding and stride = 1, the spatial dimensions remain unchanged:

- After 1st conv: 128×128  
- After 2nd conv: 128×128

So, final output = **128×128×(number of filters in 2nd layer)**  
(If not specified, keep depth abstract → **128×128×D**)

---

## Question 6

**Line removed:** `model.train()`  
This puts the model in **training mode** (affects BatchNorm, Dropout)

If you **remove** it:
- Model stays in **eval mode**
- Dropout/BacthNorm will behave **differently** (i.e., fixed stats)
- Training may be incorrect → gradients won't update as expected

🧠 **Conclusion:** Always call `model.train()` before training.

---
