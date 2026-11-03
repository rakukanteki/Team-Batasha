**# Problem Statement:**

"""

A mining base on the surface of the Martian moon Phobos has been struck by an asteroid. As a competitor, your mission is to pilot a small shuttle to transport the remaining mining crew from the surface to an orbiting Earth-return spacecraft. The only problem: the asteroid impact kicked up a dense debris field near the mining base, and with the low gravity of Phobos there are rocks and debris everywhere. Your task is to design an autonomous controller to help your shuttle navigate through the debris field and reach safety as quickly as possible.

"""



**# What the Competitors will perform:**
"""
You will program your controller using the Python API within Space Teams PRO. To help your controller respond to the dynamic environment, you will use data from an on-board radar, as well as RGB and depth images from two controllable cameras on your shuttle. You will also have estimated spacecraft position and attitude states provided by the simulation. Your task is to observe the debris field, predict and minimize collisions, and optimize your trajectory while controlling your spacecraft. Your score will be based on the time and fuel it takes to reach the interplanetary spacecraft, but watch out: a single stray asteroid could end your mission early. The simulation is provided, along with starter code that shows example usage of functions to send thruster commands and retrieve sensor data from the radar and cameras. Competitors will extend the starter code with their own obstacle avoidance and 3D path-planning algorithms, and any auxiliary functions needed to achieve the goal.

"""

**# Solution Submissions:**
"""
Competitors will record the simulation running with their control code running on their own computer. The final submission requires two consecutive runs of the simulation, using the same solution code, recorded in the same video. The score from the two runs will be averaged to get the final solution. This is a measure to avoid a sub-par solution getting lucky 1 in 100 times and having a great score. Issues like a program crash, failure to start the sim, etc are ok to discard between consecutive runs as long as the video doesn't stop rolling while the issue is resolved. Competitors can attempt the sim as many times as they want to get a better score. Competitors can update their team's submission as many times as they like before the submission deadline. The finals qualification submission is required to be eligible to win a prize in this competition. This is a "completion grade" sort of submission, meant to ensure that your team is familiar with the competition submission process and won't need to figure it out right before the final submission deadline. It also helps us find issues with enough time to address them before the competition ends.
"""

**# Submissions:**
* (Final submission only) Claimed time, fuel, and damage
* Screen-recorded video of the full simulation (1080p 30fps recommended)
* Simulation logs
  * Copy the matching sim-run folders in `/AppData/Local/SimDynamX/STPro/Logs` into a .zip file
* Full set of competitor-written Python source code and altered user System configuration inside the sim config, if any.
  * As a GitHub repository
* (Final submission only) Additional written information about your team's solution

The submission is a claim of a certain score. The competition organizers will validate top scores for consideration in the leaderboard. Final Qualification Submission Link: `https://forms.gle/jwmR4r4NY9PM5xJKA`

# **Important Things:**
- Data from onboard radar
- 2 controllable camera (RGB + Depth)
- Estimated position and altitude in SpaceTeamsPro software.
- Avoid collision, and optimize trajectory.
- Scoring will be based on the time taken by the spacecraft to reach the mothership.
- Minus point on collisions.

# **Space Craft Specs [USEFUL LINKS]:**
- https://github.com/SimDynamX/STU4-Solution/wiki/1.-vehicle-specs-and-model-details
- https://github.com/SimDynamX/STU4-Solution/wiki/2.-mission-success,-failure,-and-scoring
- https://github.com/SimDynamX/STU4-Solution/wiki/3.-competitor-python-api