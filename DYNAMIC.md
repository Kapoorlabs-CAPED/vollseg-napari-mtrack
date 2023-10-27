- **Growth Rate Calculation**:
  - First, all growth events are detected.
  - Microtubule polymerization velocity (vg [nm/s]) is assumed to follow a near-linear polynomial function, typically implemented as a 2nd or 3rd order polynomial regularized with a linear function.
  - RANSAC identifies the largest subset of consecutive time points that follow near-linear growth.
  - The software iteratively removes time points belonging to an identified growth event from the sampling set.
  - RANSAC repeats the sampling until no further growth events can be found.
  - In the example shown, RANSAC identifies multiple growth events.
  - Finally, the software fits a linear function to all inlier points of each growth event.

- **Shrink Rate Calculation**:
  - Next, RANSAC identifies events of microtubule shrinkage.
  - Microtubule depolymerization velocity (vs [nm/s]) is often significantly higher than polymerization velocity, leading to short depolymerization events.
  - The software uses a linear model limited to fast decline for the iterative RANSAC algorithm.

- **Catastrophe Frequency Calculation**:
  - Catastrophe frequency (fc [s−1]) is determined by dividing the total number of identified shrinkage events by the total time the microtubules were growing (events/time).
  - It takes into account only full growth events when calculating catastrophe frequency.

- **Rescue Frequency Calculation**:
  - Analogously, rescue frequency (fr [s−1]) is determined by dividing the total number of identified growth events by the total time the microtubule spent shrinking (events/time).
  - Distinction is made between total catastrophes (when the microtubule shrinks all the way back to the seed) and rescues by comparing the start of the new growth event to the baseline (end point of the seed).
