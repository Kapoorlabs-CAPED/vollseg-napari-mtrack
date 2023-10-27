# Growth and Shrink Rate Calculation

The growth rate and shrink rate are computed by analyzing the slope of the kymograph, which represents microtubule behavior during its growth and shrinking phases. Specifically:

- **Growth Rate**: When a microtubule is in its growth phase, the code calculates the rate at which it is extending. This rate is computed by analyzing the slope of the kymograph corresponding to the growth phase.

- **Shrink Rate**: Conversely, when a microtubule is undergoing shrinkage, the code calculates the rate at which it is depolymerizing. The shrink rate is also determined by analyzing the slope of the kymograph, but for the shrinking phase.

The calculated growth and shrink rates are then added to the appropriate event lists, where growth rates are added to the "growth_events" list, and shrink rates are added to the "shrink_events" list.

# Catastrophe and Rescue Frequency Calculation

- **Catastrophe Frequency**: When the rate of change in microtubule length (as indicated by the slope of the kymograph) is negative (i.e., the microtubule is rapidly depolymerizing), it signifies a catastrophe event. The code identifies these events and increments the catastrophe frequency (cat_frequ) to keep track of how frequently catastrophes occur.

- Additionally, the code updates the total depolymerization time during catastrophe events. This is a measure of how long microtubules spend depolymerizing during the observation.

- **Growth Event Normalization**: When the rate is non-negative (indicating a growth event), and if the kymograph index changes (suggesting a different microtubule), the code normalizes the previously calculated catastrophe frequency. This normalization is done concerning the total time of observation, providing a normalized value that characterizes the frequency of catastrophe events relative to the overall observation time.

- Similar calculations are applied to compute rescue events. When the microtubule undergoes a rescue event, the code increments the rescue frequency (res_frequ) and maintains the total depolymerization time.

- The computed rescue frequency is also normalized with respect to the total time of observation, providing a measure of how frequently rescue events occur in relation to the total observation time.
