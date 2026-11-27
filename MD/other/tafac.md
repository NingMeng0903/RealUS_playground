Trajectory-aware feedforward admittance control for contact force tracking with time-varying reference trajectory in a teleoperated robotic ultrasound system
Author links open overlay panel
Sen Li a
, 
Yufan Pi a
, Weihua Li b
, Giuseppe Carbone c d
, Kai Wu a

Show more

Add to Mendeley

Share

Cite
https://doi.org/10.1016/j.conengprac.2025.106674
Get rights and content
Full text access
Abstract
Maintaining stable and precise contact force under dynamic trajectories is critical in teleoperated robotic ultrasound systems. Traditional admittance control (AC) methods often struggle to ensure effective force tracking when the reference trajectories provided by human operators involve significant velocity and acceleration variations. To overcome these limitations, this paper proposes a Trajectory-Aware Feedforward Admittance Control (TAFAC) algorithm. TAFAC improves upon conventional admittance control by incorporating feedforward compensation of velocity and acceleration terms from the reference trajectory, along with a delay-compensation mechanism. Unlike variable admittance methods reliant on online optimization or complex environment estimation, TAFAC maintains a straightforward, model-free structure that is easy to implement. Theoretical analyses demonstrate the stability and convergence of TAFAC under dynamic trajectory inputs. Comprehensive simulation and experimental validations under constant velocity (10, 15, 20 mm/s) and constant acceleration (1.0, 1.5, 2.0 mm/s²) conditions confirm TAFAC’s superior performance, significantly reducing overshoot and steady-state errors. Experimental results show overshoot reduction exceeding 60 % and RMSE reduction of over 6.5 N compared to traditional AC, highlighting TAFAC’s effectiveness for ensuring safe, accurate, and stable robotic ultrasound operations.
Previous article in issue
Next article in issue
Keywords
Force controlAdmittance controlDynamic reference trajectoryUltrasound robot
1. Introduction
Interaction with dynamic and complex environments has made significant strides in recent years, driven by increasing demands for precision and reliability across domains such as industrial automation (Zhang et al., 2022), human-robot interaction (Chen et al., 2024), and teleoperation (Yang et al., 2024). Among these applications, teleoperated robotic ultrasound systems represent a transformative advancement in medical diagnostics, offering a promising solution to the limited availability of ultrasound services in remote and medically underserved regions (Koizumi, Warisawa, Nagoshi, Hashizume, Mitsuishi, 2009, Wu, Feng, An, Carbone, Li, 2025). These systems enable physicians to remotely manipulate an ultrasound probe via a robotic platform-often enhanced with force and haptic feedback-thereby expanding access to high-quality imaging and reducing the reliance on-site medical personnel.
One of the primary challenges in robotic ultrasound systems is maintaining a safe and stable contact force between the probe and the patient’s body. This is critical not only for ensuring patient comfort but also for acquiring diagnostically useful ultrasound images (Bao, Wang, Zheng, Housden, Hajnal, Rhode, 2023, Huang, Gao, Wang, 2024, Mathiassen, Fjellin, Glette, Hol, Elle, 2016). In teleoperated scenarios (Fig. 1), this challenge becomes even more pronounced, as the probe’s motion is governed in real time by the operator via a haptic interface. Frequent adjustments to the probe’s pose are necessary to optimize imaging quality, yet these dynamic inputs can lead to excessive or unstable contact forces (Si et al., 2024). This human-in-the-loop control introduces uncertainties and variability in trajectory dynamics that can compromise force stability.
Fig. 1
Download: Download high-res image (287KB)
Download: Download full-size image
Fig. 1. The scheme of the teleoperated robotic ultrasound system.

To address such challenges, force control methods have been widely adopted in robotic ultrasound applications. Force control in robotic systems typically employs strategies such as impedance control (Hogan, 1985), admittance control (Seraji, 1994). Admittance control, in particular, has been widely adopted in robotic ultrasound applications due to its ability to translate external force inputs into compliant motion, thereby offering a stable and intuitive force-position interaction framework (Ferraguti et al., 2019). In teleoperation, admittance controllers enable the robot to follow operator-defined trajectories while maintaining safe interaction forces with soft tissue (Santos and Cortesão, 2013). Prior studies have demonstrated the effectiveness of such methods under slowly varying or quasi-static reference inputs (Mathiassen, Fjellin, Glette, Hol, Elle, 2016, Xiao, Wang, Long, Yang, 2025).
However, in real teleoperated ultrasound procedures, reference trajectories are often dynamic and unpredictable. Generated in real time by human operators, these trajectories frequently include significant velocity and acceleration components, especially during rapid adjustments or target repositioning (Fu et al., 2025). When the probe approaches the patient’s skin, these dynamic inputs can result in excessive interaction forces at contact onset, leading to discomfort or even harm to the patient (Munir, Al-Battal, Al-Sheghri, Becher, Noga, Punithakumar, 2025, Si, Wang, Yang, 2024). Classical admittance control lacks provisions for such dynamic transitions and often fails to regulate force effectively under these conditions, which will be formally analyzed in Section 2.
Recent studies have focused on enhancing classical admittance control by making it adaptable to uncertain and variable interaction environments (Beber, Lamon, Nardi, Fontanelli, Saveriano, Palopoli, 2024, Yoon, Na, Song, 2024). For example, variable admittance control strategies are adopted to enhance compliance. Adaptive variable admittance control strategies, which dynamically adjust the virtual parameters (such as damping or stiffness) based on the force tracking error-often using adaptive laws-have been widely adopted in robotic force control (Duan, Gan, Chen, Dai, 2018, Hamedani, Sadeghian, Zekri, Sheikholeslam, Keshmiri, 2021). For instance, Xie et al. (2023) proposed an adaptive variable admittance control (AVAC) approach, where the damping parameter is adjusted online based on a PID law according to the force tracking error. This method effectively compensates for disturbances and improves both transient and steady-state force control, as demonstrated in mobile robot polishing experiments. Fu et al. (2024) proposed a variable admittance control strategy for robotic ultrasound scanning, where control parameters are optimized in real time to improve force stability during contact. While effective in complex environments, these methods neglect human input in teleoperation. Environmental information is estimated by force and position data to achieve robust and accurate force control (Roveda, Iannacci, Vicentini, Pedrocchi, Braghin, Tosatti, 2015, Zhang, Khamesee, 2017). Xiao et al. (2025) employed a Kalman filter-based estimation method within a variable admittance framework, allowing the robot to adaptively modulate its compliance based on real-time contact characteristics during ultrasound scanning. However, these strategies rely heavily on environment estimation, which introduces latency and uncertainty. Learning based methods are also popular to enable robots learning force control strategies from human experts (Ning, Chen, Zhang, Liao, 2021, Niu, Yang, Zhang, Ji, Jiang, Liu, 2025). Though these works enhance adaptability to environmental variations, they generally assume the reference trajectory is quasi-static and do not explicitly consider dynamic human-generated trajectories in teleoperation.
While the above methods contribute significantly to the development of adaptive and robust force control in ultrasound robotics, they predominantly focus on environmental uncertainty, contact phase tuning, or periodic disturbances. However, the specific challenge of real-time reference trajectory dynamics frequently encountered in teleoperated diagnostic procedures remains insufficiently addressed. This motivates the development of a novel Trajectory-Aware Feedforward Admittance Control (TAFAC) algorithm tailored for contact force tracking in such systems.
The main contributions of this study are summarized as follows:
1. The limitations of conventional admittance control in force tracking under time-varying reference trajectories are analyzed. Specifically, a transfer function between the reference trajectory and the contact force error is derived, and the final value theorem is applied to reveal the inability of conventional admittance control to guarantee accurate force tracking in such cases.
2. A novel Trajectory-Aware Feedforward Admittance Control (TAFAC) method is proposed. By incorporating feedforward and delay-compensation terms derived from dynamic reference trajectories into the admittance control law, the transfer function between the force error and the reference trajectory is reshaped to achieve stable force tracking under time-varying reference trajectories.
3. Rigorous theoretical analysis is provided for the stability and convergence of TAFAC. In particular, the Routh criterion is applied to guarantee closed-loop stability, and the convergence of the force error under constant-acceleration reference trajectories is established using the final value theorem.
4. Both simulations and experiments are conducted under constant-velocity and constant-acceleration reference trajectories. The results demonstrate that TAFAC consistently outperforms conventional admittance control (AC) and adaptive variable admittance control (AVAC) in terms of force tracking accuracy.
The remainder of this paper is organized as follows: Section 2 presents the interaction model and highlights the limitations of classical admittance control under dynamic reference trajectories. Section 3 introduces the proposed control algorithm and provides its stability analysis. Section 4 describes the simulation and experimental setup, as well as the results presented in Section 5. Section 6 analyzes the limitations of the traditional AC and the comparative method AVAC, and further provides a deeper explanation of the effectiveness of the proposed TAFAC method. Finally, Section 7 concludes the study and discusses future research directions.
2. Problem formulation
2.1. Interaction model of robot and environment
The teleoperated robotic ultrasound system used in this study is shown in Fig. 1. It mainly consists of a 6-DOF (degree of freedom) doctor-side haptic device and a 6-DOF patient-side robotic manipulator. The control scheme and contact illustration of the system are illustrated in Fig. 2. The doctor manipulates the handle of the haptic device (Fig. 2(a)), and the real-time reference trajectory command xin is sent to the robot controller for regulating the probe (Fig. 2(b)). In Fig. 2(b), xm denotes the modified output trajectory generated by the admittance controller, which serves as the command trajectory for the robot position controller. Usually, it is assumed that the position controller has no error, which means xm could be treated as the real position of the robot. When the robot is moving in free space shown in Fig. 2(b), the probe directly follows the reference trajectory xin sent from the haptic device. In this manner, the position error Δ⁢𝑥=𝑥𝑚−𝑥𝑖⁢𝑛=0, enabling trajectory tracking for the probe to follow the doctor’s command. Once the probe makes contact with the environment(contact force fe > 0), the real-time reference trajectory command xin is sent to the admittance controller. Then, the admittance controller outputs the modified trajectory xm according to the admittance control law to the robot position controller. In traditional force tracking admittance control, the control law is written as Eqs. (1) and (3).
Fig. 2
Download: Download high-res image (119KB)
Download: Download full-size image
Fig. 2. Control Scheme and contact illustration of the teleoperated ultrasound robot. (a) The doctor manipulates the Haptic Device, and the Reference Trajectory is sent to the robot. (b) The Robot moves in free space following the reference trajectory, Δ⁢𝑥=𝑥𝑚−𝑥𝑖⁢𝑛=0 (c)Robot contact with the environment under reference trajectory input.

The environment is commonly represented using the Kelvin-Voigt linear model (Haddadi and Hashtrudi-Zaad, 2008), which characterizes it as a linear ”spring-damper” system in Eq. (2). Since each Cartesian component is independent, here a one-dimensional case is considered to describe these models without loss of generality. The mathematical expressions for these models are as follows:
(1)
𝑚⁢(
¨
𝑥
𝑚−
¨
𝑥
𝑖⁢𝑛)+𝑏⁢(
˙
𝑥
𝑚−
˙
𝑥
𝑖⁢𝑛)+𝑘⁢(𝑥𝑚−𝑥𝑖⁢𝑛)=Δ⁢𝑓
(2)
𝑘𝑒⁢(𝑥𝑒−𝑥𝑚)−𝑏𝑒⁢
˙
𝑥
𝑚=𝑓𝑒
(3)
Δ⁢𝑓=𝑓𝑒−𝑓𝑑
wherem, b, and k represent the virtual mass, damping, and stiffness coefficients of the admittance controller. xin denotes the input desired trajectory, while xm is the output modified trajectory of the admittance controller according to force error Δf, which is the difference between the measured contact force fe and desired force fd. The environment is modeled by a Kelvin-Voigt linear model, where ke and be represent the stiffness and damping coefficients, respectively. xe is the environment equilibrium position.
Due to visual occlusion or network delay, the doctor may still move the handle downward with a certain velocity or acceleration even when the probe is very close to the patient’s skin. This leads to great excessive force when the probe starts to contact with the environment under the doctor’s reference trajectory input, as shown in Fig. 2(c).
Based on the above discussions, this study aims to reduce force error and achieve force tracking control under dynamic reference trajectory input for admittance control in teleoperated robotic ultrasound applications.
2.2. Limitations of traditional admittance control under dynamic reference trajectory
The control scheme for admittance control under these circumstances is illustrated in Fig. 3. The force sensor measures the contact force fe on the probe. When fe is detected to exceed 0, the reference trajectory 𝑥𝑖⁢𝑛,
˙
𝑥
𝑖⁢𝑛,
¨
𝑥
𝑖⁢𝑛 generated from the haptic device will be sent to the admittance controller, and the modified trajectory position xm is output to the robot position controller. Otherwise, when the force sensor detects no contact force, the reference trajectory is directly sent to the robot position controller for trajectory tracking.
Fig. 3
Download: Download high-res image (134KB)
Download: Download full-size image
Fig. 3. The admittance control scheme.

Traditional force tracking admittance control cannot converge the contact force to the desired level under a dynamic reference trajectory input. To improve the controller’s performance under dynamic reference trajectory input, the limitations of traditional admittance control are first theoretically investigated. Then, an improved admittance control is proposed, which will be discussed in Section 3. Here, the Steady-State Error(SSE) of traditional admittance control is derived :
Taking the Laplace transform on both sides of the Eqs. (1)–(3) yields:
(4)
(𝑚⁢𝑠2+𝑏⁢𝑠+𝑘)⁢(𝑥𝑚⁡(𝑠)−𝑥𝑖⁢𝑛⁡(𝑠))=Δ⁢𝑓⁡(𝑠)
(5)
𝑘𝑒⁢𝑥𝑒
𝑠
 
−(𝑘𝑒+𝑏𝑒⁢𝑠)⁢𝑥𝑚⁡(𝑠)=𝑓𝑒⁡(𝑠)
(6)
Δ⁢𝑓⁡(𝑠)=𝑓𝑒⁡(𝑠)−𝑓𝑑⁡(𝑠)
Given that fd(s) and xin(s) are the system inputs, and xm(s) and fe(s) are intermediate feedback outputs, Eqs. (4)-(6) can be combined, and the algebraic terms involving xm(s) and fe(s) eliminated. This yields the relationship between Δf(s), fd(s), and xin(s):
(7)
Δ⁢𝑓⁡(𝑠)=
(
𝑘𝑒⁢𝑥𝑒−𝑠⁢𝑓𝑑⁡(𝑠)
𝑘𝑒⁢𝑠+𝑏𝑒⁢𝑠2
 
−𝑥𝑖⁢𝑛⁡(𝑠))
(
1
𝑘𝑒+𝑏𝑒⁢𝑠
 
+
1
𝑚⁢𝑠2+𝑏⁢𝑠+𝑘
 
)
 
To determine the steady-state value of Δf(t), the final value theorem of the Laplace transform is applied as follows:
(8)
lim
𝑡→∞
 ⁡Δ⁢𝑓⁡(𝑡)	= 
lim
𝑠→0
 ⁡𝑠⁢Δ⁢𝑓⁡(𝑠)
= 
lim
𝑠→0
 ⁡
𝑠⁢(𝑚⁢𝑠2+𝑏⁢𝑠+𝑘)⁢(𝑘𝑒+𝑏𝑒⁢𝑠)
𝑚⁢𝑠2+(𝑏+𝑏𝑒)⁢𝑠+𝑘+𝑘𝑒
 
×(
𝑘𝑒⁢𝑥𝑒−𝑠⁢𝑓𝑑⁡(𝑠)
𝑘𝑒⁢𝑠+𝑏𝑒⁢𝑠2
 
−𝑥𝑖⁢𝑛⁡(𝑠))
= 
lim
𝑠→0
 ⁡
𝑘⁢(𝑘𝑒⁢𝑥𝑒−𝑠⁢𝑓𝑑⁡(𝑠))
𝑘+𝑘𝑒
 
−
(𝑚⁢𝑠2+𝑏⁢𝑠+𝑘)⁢(𝑘𝑒⁢𝑠+𝑏𝑒⁢𝑠2)⁢𝑥𝑖⁢𝑛⁡(𝑠)
𝑘+𝑘𝑒
 
 
Generally, the desired contact force fd(t) is known, so fd(s) is a known function of s. According to the final value theorem:
(9)
lim
𝑠→0
 ⁡𝑠⁢𝑓𝑑⁡(𝑠)=𝑓𝑑⁢𝑛
where fdn represents the final desired force, which is known.
Consider the situation where the motion of the robot’s end-effector is static, i.e., satisfying 
˙
𝑥
𝑖⁢𝑛=0 and 𝑥𝑖⁢𝑛=𝜆(here λ is a constant), then 𝑥𝑖⁢𝑛⁡(𝑠)=
𝜆
𝑠
 
. Substituting this into Eq. (8) and setting its steady-state value to zero yields:
(10)
lim
𝑡→∞
 ⁡Δ⁢𝑓⁡(𝑡)	= 
lim
𝑠→0
 ⁡
𝑘⁢(𝑘𝑒⁢𝑥𝑒−𝑠⁢𝑓𝑑⁡(𝑠))
𝑘+𝑘𝑒
 
−
𝜆⁢𝑘⁢𝑘𝑒
𝑘+𝑘𝑒
 
= 
lim
𝑠→0
 ⁡
𝑘
𝑘+𝑘𝑒
 
⁢(𝑘𝑒⁢(𝑥𝑒−𝜆)−𝑓𝑑⁢𝑛)=0
 
From a mathematical perspective, the conditions for satisfying the steady-state value lim𝑡→∞⁡Δ⁢𝑓⁡(𝑡)=0 are as follows: 1.𝑘=0 or 2.𝜆=𝑥𝑒−
𝑓𝑑⁢𝑛
𝑘𝑒
 
For condition 1, the literature Jung et al. (2004) has studied setting the stiffness coefficient k to zero to achieve ideal force tracking control. For condition 2, if the stiffness of the external environment ke and the precise initial position xe are known, the robot end-effector position λ can be calculated to meet the desired force requirements. However, as mentioned earlier, it is challenging to obtain these values directly through compensation or estimation. Even slight compensation errors can lead to the accumulation of control errors, resulting in increasingly large steady-state force errors. Therefore, condition 1 is often used in static force tracking control. It can be concluded that when the environment is unknown, and the initial trajectory of the robot end-effector undergoes dynamic changes, the necessary condition for zero steady-state force error is 𝑘=0. Under this condition, Eq. (4) is simplified as:
(11)
𝑚⁢(
¨
𝑥
𝑚−
¨
𝑥
𝑖⁢𝑛)+𝑏⁢(
˙
𝑥
𝑚−
˙
𝑥
𝑖⁢𝑛)=Δ⁢𝑓
Thus, Eq. (8) can be simplified to:
(12)
lim
𝑡→∞
 ⁡Δ⁢𝑓⁡(𝑡)= 
lim
𝑠→0
 ⁡𝑠⁢Δ⁢𝑓⁡(𝑠)
= 
lim
𝑠→0
 −
(𝑚⁢𝑠2+𝑏⁢𝑠)⁢(𝑘𝑒⁢𝑠+𝑏𝑒⁢𝑠2)⁢𝑥𝑖⁢𝑛⁡(𝑠)
𝑚⁢𝑠2+(𝑏+𝑏𝑒)⁢𝑠+𝑘𝑒
 
= 
lim
𝑠→0
 −
𝑚⁢𝑏𝑒⁢𝑠4+(𝑚⁢𝑘𝑒+𝑏⁢𝑏𝑒)⁢𝑠3+𝑏⁢𝑘𝑒⁢𝑠2
𝑚⁢𝑠2+(𝑏+𝑏𝑒)⁢𝑠+𝑘𝑒
 
⁢𝑥𝑖⁢𝑛⁡(𝑠)
 
In a teleoperated robotic ultrasound system, the reference trajectory xin of the robot end-effector is decided by the doctor’s manipulation, meaning 
˙
𝑥
𝑖⁢𝑛≠0,
¨
𝑥
𝑖⁢𝑛≠0. Considering that the haptic device is operated manually and the input trajectory is often pre-processed with low-pass filtering, the acceleration of the commanded motion can be assumed to be approximately constant over short time intervals. From a modeling and control design perspective, assuming a constant desired acceleration simplifies the derivation of the system’s dynamics and facilitates analytical stability analysis, particularly in evaluating the fundamental limitations of admittance control in force tracking under dynamic references. Therefore, setting the reference acceleration 
¨
𝑥
𝑖⁢𝑛 as constant is a physically reasonable, engineering-practical, and theoretically justified approximation within the scope of the considered application. Assuming 𝑥𝑖⁢𝑛⁡(𝑡)=𝜆⁢𝑡2, which implies 𝑥𝑖⁢𝑛⁡(𝑠)=
2⁢𝜆
𝑠3
 
. Substituting this into Eq. (12) gives:
(13)
lim
𝑡→∞
 ⁡Δ⁢𝑓⁡(𝑡)= 
lim
𝑠→0
 −
𝑚⁢𝑏𝑒⁢𝑠3+(𝑚⁢𝑘𝑒+𝑏⁢𝑏𝑒)⁢𝑠2+𝑏⁢𝑘𝑒⁢𝑠
𝑠2
 
·
2⁢𝜆
𝑘𝑒
 
 
Clearly, Eq. (13) diverges to infinity. This observation demonstrates that constant admittance control inevitably results in steady-state errors under the dynamic reference trajectory of the robot end-effector. Moreover, as the motion rate increases (i.e., as λ becomes larger), the steady-state error also grows significantly. This highlights the inherent limitations of traditional admittance control. The following section proposes a novel adaptive admittance control algorithm to reduce steady-state errors and enhance the robot’s active compliance during dynamic motion. The stability and convergence of the proposed algorithm will be analyzed and verified in the following section.
3. Improved admittance control under dynamic reference trajectory
3.1. Improved admittance control algorithm
As described above, traditional admittance control does not account for the dynamic motion of the model, leading to a final steady-state deviation. To enhance the applicability of force-position coordination control, this section improves upon the traditional admittance control law by incorporating the motion state of the robot’s end-effector.
Specifically, the mathematical model for the end-effector in Eq. (1) is modified as follows:
(14)
𝑚⁢(
¨
𝑥
𝑚⁡(𝑡)−
¨
𝑥
𝑖⁢𝑛⁡(𝑡))+𝑏⁢(
˙
𝑥
𝑚⁡(𝑡−𝜏)−
˙
𝑥
𝑖⁢𝑛⁡(𝑡−𝜏))
=Δ⁢𝑓⁡(𝑡−𝜏)−𝑏⁢
˙
𝑥
𝑖⁢𝑛⁡(𝑡)−(𝑚−𝑏⁢𝜏)⁢
¨
𝑥
𝑖⁢𝑛⁡(𝑡)
 
Where τ represents the system’s control cycle. Compared to traditional admittance control strategies, the proposed control strategy integrates the desired trajectory velocity and acceleration of the robot’s end-effector, as well as the overall system’s control frequency. This approach effectively addresses the limitations discussed earlier. To apply the control law to real robot systems, Eq. (14) is appropriately modified into a discrete form as follows:
(15)
Δ⁢
¨
𝑥
⁡[𝑘]=
1
𝑚
 
⁢[Δ⁢𝑓⁡[𝑘−1]−𝑏⁢
˙
𝑥
𝑖⁢𝑛⁡[𝑘]−(𝑚−𝑏⁢𝑇𝑠)⁢
¨
𝑥
𝑖⁢𝑛⁡[𝑘]−𝑏⁢Δ⁢𝑥⁡[𝑘−1]]
 
Where Δ⁢𝑥=𝑥𝑚−𝑥𝑖⁢𝑛 is the position error. k and Ts are the sample index and sampling period of the real robot system. The control framework of the proposed adaptive admittance control system is shown in Fig. 4. The following section will provide proof of the stability and convergence of the proposed control strategy.
Fig. 4
Download: Download high-res image (89KB)
Download: Download full-size image
Fig. 4. The control scheme of the proposed algorithm.

3.2. Stability and convergence of adaptive control strategy
Theorem 1. The closed-loop system under the proposed Trajectory-Aware Feedforward Admittance Control (TAFAC) is asymptotically stable if the system parameters satisfy the following condition:
(16)
𝜏<min⁡(
𝑚
𝑏+𝑏𝑒
 
,
𝑏+𝑏𝑒
𝑘𝑒
 
)
Proof: Based on the delay property of the Laplace transform, one has:
(17)
ℒ⁡[𝐹⁡(𝑡−𝜏)]=𝑒−𝜏⁢𝑠⁢𝐹⁡(𝑠)
Applying the above property to both sides of Eq. (14) and taking the Laplace transform yields:
(18)
(𝑚⁢𝑠2+𝑏⁢𝑠⁢𝑒−𝜏⁢𝑠)⁢(𝑥𝑚⁡(𝑠)−𝑥𝑖⁢𝑛⁡(𝑠))=𝑒−𝜏⁢𝑠⁢Δ⁢𝑓⁡(𝑠)−(𝑏⁢𝑠+(𝑚−𝑏⁢𝜏)⁢𝑠2)⁢𝑥𝑖⁢𝑛⁡(𝑠)
 
Assuming the desired force fd and the initial position of the environment xe are constants, both are set to zero for simplification. This yields the relationship between the reference trajectory relationship between xm(s) and Δf(s) can be expressed as:
(19)
Δ⁢𝑓⁡(𝑠)=−(𝑘𝑒+𝑏𝑒⁢𝑠)⁢𝑥𝑚⁡(𝑠)
Substituting Eq. (19) into Eq. (18) yields:
(20)
Δ⁢𝑓⁡(𝑠)=
𝑏⁢𝑠⁢(1−𝑒−𝜏⁢𝑠)−𝑏⁢𝜏⁢𝑠2
𝑒−𝜏⁢𝑠+
𝑚⁢𝑠2+𝑏⁢𝑠⁢𝑒−𝜏⁢𝑠
𝑘𝑒+𝑏𝑒⁢𝑠
 
 
⁢𝑥𝑖⁢𝑛⁡(𝑠)
Expanding 1−𝑒−𝜏⁢𝑠 using the Taylor series gives:
(21)
1−𝑒−𝜏⁢𝑠=𝜏⁢𝑠−
(𝜏⁢𝑠)2
2!
 
+
(𝜏⁢𝑠)3
3!
 
+⋯+(−1)𝑛+1⁢
(𝜏⁢𝑠)𝑛
𝑛!
 
When the control cycle is sufficiently small, 𝑒−𝜏⁢𝑠≈1−𝜏⁢𝑠, Substituting this approximation into Eq. (20) results in :
(22)
Δ⁢𝑓⁡(𝑠)
𝑥𝑖⁢𝑛⁡(𝑠)
 
=
(𝑘𝑒+𝑏𝑒⁢𝑠)⁢𝑏⁢𝑠·𝐺
(𝑚−𝜏⁢𝑏−𝜏⁢𝑏𝑒)⁢𝑠2+(𝑏+𝑏𝑒−𝑘𝑒⁢𝜏)⁢𝑠+𝑘𝑒
 
 
where 𝐺=−
(𝜏⁢𝑠)2
2!
 
+
(𝜏⁢𝑠)3
3!
 
+⋯+(−1)𝑛+1⁢
(𝜏⁢𝑠)𝑛
𝑛!
 
. The corresponding characteristic equation of the linear system is:
(23)
(𝑚−𝜏⁢𝑏−𝜏⁢𝑏𝑒)⁢𝑠2+(𝑏+𝑏𝑒−𝑘𝑒⁢𝜏)⁢𝑠+𝑘𝑒=0
Using the Routh-Hurwitz criterion, the sufficient conditions for stability are:
(24)
⎧
{ {
⎨
{ {
⎩
𝑚−𝜏⁢𝑏−𝜏⁢𝑏𝑒	>0
𝑏+𝑏𝑒−𝜏⁢𝑘𝑒	>0
𝑘𝑒	>0
 ⟹𝜏<min⁡(
𝑚
𝑏+𝑏𝑒
 
,
𝑏+𝑏𝑒
𝑘𝑒
 
)
This completes the proof of Theorem 1, demonstrating that the proposed Trajectory-Aware Feedforward Admittance Control (TAFAC) ensures asymptotic stability of the closed-loop system provided that the control cycle satisfies the condition 𝜏<min⁡(𝑚/(𝑏+𝑏𝑒),(𝑏+𝑏𝑒)/𝑘𝑒).
Theorem 2: Given a reference trajectory with a constant acceleration, the force tracking error Δf converges asymptotically to 0 under the closed-loop system with TAFAC. Proof: Assuming 𝑥𝑖⁢𝑛⁡(𝑠)=
2⁢𝜆
𝑠3
 
, where λ is a constant, the convergence is analyzed using the terminal value theorem of Laplace Transform:
(25)
lim
𝑡→∞
 ⁡Δ⁢𝑓⁡(𝑡)= 
lim
𝑠→0
 ⁡𝑠⁢Δ⁢𝑓⁡(𝑠)
Which leads to:
(26)
lim
𝑡→∞
 ⁡Δ⁢𝑓⁡(𝑡)= 
lim
𝑠→0
 ⁡𝑠⁢Δ⁢𝑓⁡(𝑠)
= 
lim
𝑠→0
 ⁡
𝑠⁢(𝑘𝑒+𝑏𝑒⁢𝑠)⁢𝑏⁢𝑠·𝐺
(𝑚−𝜏⁢𝑏−𝜏⁢𝑏𝑒)⁢𝑠2+(𝑏+𝑏𝑒−𝑘𝑒⁢𝜏)⁢𝑠+𝑘𝑒
 
·
2⁢𝜆
𝑠3
 
=0
 
This completes the proof of Theorem 2, confirming that the force tracking error converges asymptotically to zero under TAFAC when the reference trajectory has constant acceleration.
4. Simulation and experiments
Simulations and experiments are conducted to verify the advantages of the proposed algorithm against traditional admittance control under dynamic reference trajectories.
4.1. Simulation validation
A series of robot-environment interaction simulations is designed using MATLAB Simulink. These simulations are conducted to evaluate the target force tracking performance of the proposed method compared to the traditional admittance control under dynamic reference trajectories. The simulations focus on inputting the desired force fd and the reference trajectory xin of the robot’s end-effector, while the outputs include the feedback force fe from the environmental model and the corrected trajectory xm generated by the algorithm.
The environmental model is based on the Kelvin-Voigt linear model described in Eq. (2). The simulation includes three control strategies for comparison:
(i) AC: Traditional admittance control algorithm, implemented based on Eq. (11);
(ii) TAFAC: The proposed TAFAC algorithm, implemented according to Eq. (14);
(iii) AVAC: An adaptive variable admittance controller based on PID (Xie et al., 2023), which regulates the damping coefficient according to a PID law based on force error. The equation of this method is described by:
(27)
Δ⁢𝑓⁡(𝑡)=𝑚Δ⁢
¨
𝑥
⁡(𝑡)+(𝑏+Δ⁢𝑏⁡(𝑡))⁢Δ⁢
˙
𝑥
⁡(𝑡)
 
(28)
Δ⁢𝑏⁡(𝑡)=−
𝑘𝑝Δ⁢𝑓⁡(𝑡)+𝑘𝑑Δ⁢
˙
𝑓
⁡(𝑡)+𝑘𝑖⁢∫𝑡
0Δ⁢𝑓⁡(𝑡)𝑑𝑡
Δ⁢
˙
𝑥
⁡(𝑡)
 
 
This AVAC controller is commonly used in contact force control and serves as a benchmark for evaluating the performance of the proposed method. For AVAC in particular, the PID parameters (𝑘𝑝=0.1,𝑘𝑖=2,𝑘𝑑=0.01) are adopted. These values were obtained through a systematic tuning process. Specifically, each parameter was swept within a predefined range based on prior experience and stability constraints:
- kp was varied from 0.05 to 0.5 in increments of 0.05,
- ki was varied from 0.5 to 5 in increments of 0.5,
- kd was varied from 0.005 to 0.05 in increments of 0.005.
The optimal set of parameters was selected based on a multi-criteria evaluation including convergence speed, overshoot, and steady-state error. The selected parameters represent the best-performing configuration among all tested combinations. For simplicity, the simulation does not consider any external disturbances or account for errors in the robot’s position controller. The parameters in the simulation are configured as follows in Table 1.
Table 1. Simulation parameters.

Symbol	Description	Value
m	Virtual Mass	6 kg
b	Virtual Damping	900 N  ·  s/m
xe	Environment Initial Position	0 m
ke	Environment Stiffness	3000 N/m
be	Environment Damping	40 N · s/m
τ	Control cycle	2×10−3 s
T	Simulation Time	10 s
The parameters of the admittance controller m and b are well-tuned by experience. The average stiffnesses are 3237 N/m and 1489 N/m for the stiff and soft phantom, respectively, in Jiang et al. (2021). ke =3000 N/m is chosen to simulate the ultrasound inspection condition. The stiffness and damping coefficients are set to 3000 N/m and 40 N  ·  s/m, respectively. The control cycle is set to 2 ms, which corresponds with the actual UR5e robot control cycle used in the experiment. The simulation is implemented in a discrete-time model, where the control cycle serves as the sampling time (i.e., the simulation step size).
The simulation focuses on the contact force tracking performance during dynamic reference trajectory inputs. Two types of simulations are conducted, simulating the ultrasound probe making contact with the environment at a constant reference velocity and at a constant acceleration. In the simulation scenario, the probe starts from a 30 mm distance above the environment with a constant velocity or acceleration to move towards the environment. The two types of simulation tests are described as follows: Constant Velocity (CV): The desired contact force is set to fd=15 N. The reference trajectories are set as 𝑥𝑖⁢𝑛⁡(𝑡)=𝑣𝑖⁢𝑛⁢𝑡, where reference velocity 𝑣𝑖⁢𝑛=
˙
𝑥
𝑖⁢𝑛 is tested with [10, 15, 20] mm/s in the constant velocity simulation.
Constant Acceleration (CA): The desired contact force is set to fd=15 N. The reference trajectory is 
¨
𝑥
𝑖⁢𝑛⁡(𝑡)=
1
2
 
⁢𝑎𝑖⁢𝑛⁢𝑡2, where reference acceleration 𝑎𝑖⁢𝑛=
¨
𝑥
𝑖⁢𝑛 is tested with [1, 1.5, 2] mm/s² in the constant acceleration simulation. Here xe=-30 mm is simulated by setting a corresponding initial velocity at 𝑡=0.
Fig. 5 presents simulation results for the contact force tracking performance under both constant velocity (a-c) and constant acceleration (d-f) reference trajectories. Three control methods-traditional admittance control (AC), adaptive variable admittance control (AVAC), and the proposed TAFAC-are comparatively analyzed.
Fig. 5
Download: Download high-res image (367KB)
Download: Download full-size image
Fig. 5. Simulation results of constant velocity (a–c) and constant acceleration (d–f).

In the constant velocity scenarios (Fig. 5(a–c)), the AC method consistently demonstrates substantial overshoot and evident steady-state errors, significantly deviating from the desired force trajectory. Although the AVAC method effectively reduces transient fluctuations compared to AC, it still exhibits notable overshoot and requires relatively longer durations to reach steady-state. Conversely, the TAFAC method achieves rapid convergence, minimal overshoot, and stable force tracking at all velocities, maintaining excellent alignment with the desired contact force trajectory throughout the simulation duration.
In constant acceleration scenarios (Fig. 5(d–f)), the inadequacies of AC become even more pronounced, exhibiting divergent force trajectories at higher accelerations. AVAC method demonstrates improved transient convergence compared to AC; however, as highlighted by the enlarged inset figures, it consistently exhibits observable steady-state errors, indicating limitations in accurately maintaining the desired force under dynamic conditions. Conversely, TAFAC consistently maintains minimal overshoot and achieves stable, precise steady-state tracking. The enlarged inset figures further illustrate TAFAC’s capability in eliminating steady-state deviations, clearly distinguishing its superior performance relative to AVAC.
In summary, simulation results demonstrate the effectiveness of the proposed TAFAC method, clearly indicating that the proposed TAFAC method provides superior performance across various dynamic conditions. Specifically, TAFAC delivers excellent transient response, significantly reduced overshoot, and accurate steady-state tracking, outperforming both traditional AC and AVAC methods. These results further confirm the effectiveness and robustness of TAFAC for robot-environment interaction tasks under diverse dynamic conditions.
4.2. Experimental validation
According to the simulation results, the same two types of contact experiments are conducted in a real robotic system.
4.2.1. Experiment setup
This experiment utilizes a 6-DOF UR5e robotic manipulator equipped with an ultrasound (US) probe (DAX, Siemens Healthineers), employing the robot’s internal force/ torque sensor for data acquisition. The UR5e robotic manipulator has a payload capacity of 5 kg, a repeatability of  ± 0.03 mm, and a working radius of 550 mm. The specific parameters of the force/torque sensor are listed in Table 2. The contact environment consists of a rubber block measuring 150 × 150 × 40 mm, with an estimated contact stiffness of approximately 3000 N/m between the probe and the rubber surface. This stiffness value closely approximates the actual conditions encountered in ultrasound inspection. Communication between the controller and the robot is established via a TCP connection over an Ethernet cable. Both the robotic system and the force sensor operate at a sampling frequency of 500 Hz.
Table 2. Parameters of force/torque sensor.

Empty Cell	Force, x-y-z	Torque, x-y-z
Range	±50.0 N	±10.0 N · m
Precision	±3.5 N	±0.2 N · m
Accuracy	±4.0 N	±0.3 N · m
The computational unit used in this experiment is equipped with an Intel(R) Core(TM) i5-10500T CPU running at 2.30 GHz. Algorithms of three comparison methods (AC, TAFAC, and AVAC) are implemented within the motion planning framework of the robot. The programs are developed in Python, using the ur_rtde interface to enable real-time control of the UR5e robotic manipulator. The experimental setup is depicted in Fig. 6. The same admittance parameters are adopted for all three comparison methods, and are consistent with the parameters in the simulation. For AVAC, the PID parameters are consistent with those used in the simulation.
Fig. 6
Download: Download high-res image (203KB)
Download: Download full-size image
Fig. 6. Experimental platform.

This configuration ensures high-frequency control for real-time experimental execution, enabling accurate evaluation of the proposed and traditional control algorithms in dynamic interaction scenarios.
4.2.2. Experiment design
In this experiment, the robotic end-effector starts from a fixed initial position and moves downward along the Z-axis with varying dynamic reference trajectories to achieve a target contact force. The initial fixed position is set 30 mm above the contact surface, and the target contact force is defined as 𝑓𝑑=15𝑁. Each experimental condition is conducted three times to ensure repeatability and reliability. The specific experimental conditions are as follows:
Constant Velocity (CV): In the constant velocity experiment, the robot’s end-effector moves downward following reference trajectories defined as 𝑥𝑖⁢𝑛⁡(𝑡)=𝑣𝑖⁢𝑛⁡(𝑡), where the reference velocity 𝑣𝑖⁢𝑛=
˙
𝑥
𝑖⁢𝑛 is tested with [10, 15, 20] mm/s.
Constant Acceleration (CA): In the constant acceleration experiment, the robot’s end-effector moves downward following reference trajectories defined as 𝑥𝑖⁢𝑛⁡(𝑡)=
1
2
 
⁢𝑎𝑖⁢𝑛⁢𝑡2, where the reference acceleration 𝑎𝑖⁢𝑛=
¨
𝑥
𝑖⁢𝑛 is tested with [1, 1.5, 2] mm/s.
4.2.3. Performance metrics
In this study, the Overshoot and Root Mean Square Error (RMSE) are used to evaluate the performance of the proposed and traditional algorithms under stabilized contact force conditions. The Overshoot of fe in each test is calculated by:
(29)
Overshoot=
max
𝑡
 ⁡𝑓𝑡
𝑒−𝑓𝑑
𝑓𝑑
 
×100%
where fd denotes the desired contact force, and 𝑓𝑡
𝑒 is the contact force measured at time step t. Additionally, the RMSE of the contact force is calculated according to:
(30)
RMSE=√
1
𝑇
 
⁢∑𝑇
𝑡=0(𝑓𝑡
𝑒−𝑓𝑑)2
Here, T represents the total number of sampling steps (5000 in this paper), and the time gap between each step is 2 ms. Moreover, the Settling Time (ST) is defined as:
(31)
𝑡𝑠=min⁡{𝑡|⁢∣𝑓𝑡
𝑒−𝑓𝑑∣≤𝜖,∀𝑡>𝑡𝑠}
In this experiment, ϵ is set to 5 %fd. To minimize the influence of noise from the force sensor, all experimental data are processed using a 50 Hz low-pass filter.
5. Experimental results
All experimental data are sampled at a frequency of 500 Hz and processed using a 50 Hz low-pass filter to minimize noise from the force sensor. The contact force data are analyzed from the point of initial contact, defined as the moment when the contact force reached 0.7 N. Performance metrics, including Overshoot, RMSE, and Settling Time, are calculated based on the methods described in Section 4.
Constant Velocity Results
Fig. 7 (a–c) illustrates the experimental results for the contact force tracking performance of three different admittance control methods (AC, AVAC, and the proposed TAFAC) under constant reference trajectory velocities of 10 mm/s, 15 mm/s, and 20 mm/s. To quantitatively evaluate and compare the methods, Table 3 summarizes the calculated performance metrics, including settling time (ST), overshoot, and root mean square error (RMSE) based on the entire contact force trajectory.
Fig. 7
Download: Download high-res image (460KB)
Download: Download full-size image
Fig. 7. Experiment results of constant velocity (a–c) and constant acceleration (d–f).

Table 3. Experiment results of constant velocity.

vin	10 mm/s	15 mm/s	20 mm/s
ST (s)	AC	0.95	0.84	0.75
AVAC	2.08	1.97	1.95
TAFAC	1.12	0.97	0.96
Overshoot	AC	61 %	91 %	122 %
AVAC	24 %	23 %	25 %
TAFAC	1 %	1 %	1 %
RMSE (N)	AC	8.94	13.29	17.69
AVAC	2.70	2.32	2.03
TAFAC	2.38	2.03	2.08
From Fig. 7(a–c), the traditional AC method exhibits significant overshoot and large steady-state errors (SSE) at all velocities, with the overshoot becoming more pronounced as velocity increases (61 % at 10 mm/s, rising to 122 % at 20 mm/s). In comparison, the AVAC method substantially reduces overshoot and SSE, but still exhibits notable deviations from the desired force during transient stages. The proposed TAFAC, however, achieves an exceptionally low overshoot (consistently 1 %) across all velocities and closely tracks the desired force trajectory with minimal fluctuation.
The quantitative metrics in Table 3 further support these observations. Despite AC demonstrating the shortest settling times (0.75-0.95 s), its large overshoot and RMSE indicate poor overall force tracking capability. AVAC provides improved RMSE compared to AC at higher velocities (2.03 N at 20 mm/s), but its longer settling times (2 s) and significant overshoot diminish overall control performance. The TAFAC method consistently maintains competitive settling time (1 s), minimal overshoot (1 %), and low RMSE (2.03-2.38 N), demonstrating superior combined transient and steady-state performance. Although at the highest velocity (20 mm/s), TAFAC’s RMSE (2.08 N) is slightly higher than AVAC’ts (2.03 N), the difference is negligible, and the advantage in significantly reduced overshoot and faster settling time makes TAFAC preferable overall.
In summary, these experimental results validate that TAFAC effectively balances rapid convergence, minimal overshoot, and robust tracking precision, clearly outperforming AC and presenting notable advantages over AVAC, especially in scenarios demanding both high accuracy and transient stability.
Constant Acceleration Results
The experimental results under constant acceleration conditions are depicted in Fig. 7(d–f), with quantitative comparisons summarized in Table 4. Here, the reference trajectory accelerations were set to 1.0 mm/s², 1.5 mm/s², and 2.0 mm/s², respectively. Notably, the traditional AC method failed to achieve stable force tracking under these accelerated conditions, resulting in divergence of the contact force. Moreover, as the reference acceleration increases, the divergence of the AC method occurs more rapidly and with greater magnitude, making it entirely unsuitable for dynamic force tracking in these scenarios. Therefore, its performance metrics are omitted from Table 4.
Table 4. Experiment results of constant acceleration.

ain	1 mm/s2	1.5 mm/s2	2 mm/s2
ST (s)	AVAC	2.18	2.17	2.21
TAFAC	1.10	1.09	1.26
Overshoot	AVAC	29 %	31 %	32 %
TAFAC	1 %	1 %	1 %
RMSE (N)	AVAC	3.60	3.64	3.67
TAFAC	2.4	2.4	2.6
From Fig. 7(d–f), AVAC demonstrates improved stability and convergence compared to AC; however, it still suffers from noticeable overshoot (approximately 30 %) and relatively slow convergence, as evidenced by prolonged settling times of around 2.2 s. Importantly, the locally enlarged plots in Fig. 7(d–f) further reveal that AVAC exhibits clear steady-state errors after initial convergence, indicating its limitations in maintaining precise force tracking under dynamic acceleration trajectories. In contrast, the proposed TAFAC method consistently achieves significantly faster convergence, greatly reduced overshoot, and more precise steady-state tracking performance at all acceleration levels. Specifically, TAFAC maintains overshoot at only 1 % and achieves settling times approximately half those of AVAC, ranging from 1.09 to 1.26 s.
Quantitative analysis presented in Table 4 further highlights these advantages. TAFAC achieves notably lower RMSE values (ranging from 2.4 N to 2.6 N), clearly outperforming AVAC (ranging from 3.60 to 3.67 N). The differences are substantial and consistently in favor of TAFAC, reinforcing its superior capability in tracking dynamic trajectories with varying accelerations.
Overall, these results clearly illustrate that the TAFAC method not only maintains excellent robustness against trajectory disturbances caused by increasing accelerations but also significantly outperforms the AVAC method in terms of rapid convergence, transient stability, and steady-state tracking accuracy. The observed steady-state errors of AVAC, as highlighted in the zoomed-in view, further underscore the effectiveness of the proposed control approach for practical robot-environment interactions in dynamically changing scenarios.
6. Discussion
The simulation and experimental results demonstrate that the conventional admittance control (AC) fails to regulate the contact force when the reference trajectory xin has nonzero derivatives. Specifically, a constant velocity input leads to a steady-state force error, and higher-order derivatives (e.g., constant acceleration) cause divergence, which is consistent with the analysis in Section 2.2
The adaptive variable admittance control (AVAC), which can be regarded as a PID-based feedback approach, improves force tracking but still suffers from overshoot due to the feedforward effect of xin on the contact force error. It is worth noting that although smaller ki could reduce the level of overshoot, the steady-state error also increases as the final value of force error is  
lim
𝑡→∞
 ⁡Δ⁢𝑓⁡(𝑡)=𝑎0⁢
𝑏𝑒
𝑘𝑖
 
 (see Appendix A for the derivation process). Although overshoot is observed, AVAC significantly enhances the regulation compared with AC. With constant velocity references, AVAC stabilizes the force after overshoot; with constant acceleration, it achieves stability but introduces a steady-state error that increases with acceleration magnitude. Increasing the derivative gain can reduce overshoot to some extent, but excessive values amplify sensor noise and risk destabilizing the system.
In contrast, the proposed TAFAC method incorporates a feedforward compensation of the reference trajectory, which effectively increases the relative degree of the transfer function between the reference trajectory xin and the force error Δf. This allows the contact force to remain stable and convergent even when the reference trajectory involves higher-order derivatives, thereby overcoming the limitations of both AC and AVAC.
7. Conclusion
This paper introduces the Trajectory-Aware Feedforward Admittance Control (TAFAC) method, which addresses the inherent limitations of traditional admittance control in managing dynamic reference trajectories commonly encountered in teleoperated robotic ultrasound procedures. By incorporating trajectory feedforward and delay compensation, the proposed method effectively reduces overshoot and enhances force tracking accuracy. Stability and convergence of TAFAC were theoretically proven, and its effectiveness was rigorously validated through simulations and real-robot experiments. Results consistently indicated superior transient and steady-state performances, significantly outperforming both traditional AC and adaptive variable admittance control (AVAC).
It should be noted that the present work primarily evaluates TAFAC under typical constant velocity and constant acceleration trajectories, which, although representative, do not encompass the full diversity of human-operated reference trajectories encountered in practical teleoperation, such as non-stationary, abrupt, or highly stochastic movements. Future research will focus on extending and validating TAFAC in more complex and realistic scenarios, including variable environmental stiffness and diverse human-generated trajectory profiles, to further enhance its applicability and robustness for real-world human-robot interaction tasks.
CRediT authorship contribution statement
Sen Li: Writing – review & editing, Software, Methodology, Conceptualization. Yufan Pi: Writing – original draft, Methodology, Conceptualization. Weihua Li: Validation, Writing – review & editing. Giuseppe Carbone: Validation, Visualization, Writing – review & editing. Kai Wu: Writing – review & editing, Supervision, Funding acquisition.
Declaration of competing interest
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.
Acknowledgements
This work was supported by the GJYC program of Guangzhou [grant number 2024D03J0005]; and the Fundamental Research Funds for the Central Universities [grant number 2024ZYGXZR107]
Appendix A. Appendix
The AVAC control law could be written as:
(A.1)
𝑚⁢
¨
𝑒
⁡(𝑡)+𝑏⁢
˙
𝑒
⁡(𝑡)=Δ⁢𝑓⁡(𝑡)+𝑘𝑝⁢Δ⁢𝑓⁡(𝑡)+𝑘𝑖⁢∫𝑡
0Δ⁢𝑓⁡(𝑡)𝑑𝑡+𝑘𝑑⁢Δ⁢
˙
𝑓
⁡(𝑡)
Taking Laplacian transformation of Eq. (A.1), one has:
(A.2)
𝑚⁢𝑠2⁢𝐸⁡(𝑠)+𝑏⁢𝑠⁢𝐸⁡(𝑠)=(1+𝑘𝑝)⁢Δ⁢𝐹⁡(𝑠)+𝑘𝑑⁢𝑠⁢Δ⁢𝐹⁡(𝑠)+𝑘𝑖⁢
1
𝑠
 
⁢Δ⁢𝐹⁡(𝑠)
Factorization on both sides, one has:
(A.3)
(𝑚⁢𝑠2+𝑏⁢𝑠)⁢𝐸⁡(𝑠)=((1+𝑘𝑝)+𝑘𝑑⁢𝑠+
𝑘𝑖
𝑠
 
)⁢Δ⁢𝐹⁡(𝑠)
Where 𝑒⁡(𝑡)=𝑥𝑚⁡(𝑡)−𝑥𝑖⁢𝑛⁡(𝑡), thus 𝐸⁡(𝑠)=𝑋𝑚⁡(𝑠)−𝑋𝑖⁢𝑛⁡(𝑠). Submitting this into Eq. (A.3), one has:
(A.4)
(𝑚⁢𝑠2+𝑏⁢𝑠)⁢(𝑋𝑚⁡(𝑠)−𝑋𝑖⁢𝑛⁡(𝑠))=((1+𝑘𝑝)+𝑘𝑑⁢𝑠+
𝑘𝑖
𝑠
 
)⁢Δ⁢𝐹⁡(𝑠)
Now the relationship between Xin(s) and ΔF(s) is established. However, Xm(s) is an intermediate variable and needs to be eliminated. According to the assumption that there is no error between the real robot position x and the commanded position xm, which means 𝑥=𝑥𝑚. Then the environment model becomes:
(A.5)
𝑓𝑒⁡(𝑡)=−𝑘𝑒⁢(𝑥𝑚⁡(𝑡)−𝑥𝑒)−𝑏𝑒⁢
˙
𝑥
𝑚⁡(𝑡)
Normally, xe and fd are constant values, both are set to 0 for simplification, one has:
(A.6)
Δ⁢𝑓⁡(𝑡)=𝑓𝑒⁡(𝑡)=−𝑘𝑒⁢𝑥𝑚⁡(𝑡)−𝑏𝑒⁢
˙
𝑥
𝑚⁡(𝑡)
Taking Laplacian transformation of Eq. (A.6), one has the relationship between ΔF(s) and Xm(s):
(A.7)
Δ⁢𝐹⁡(𝑠)=−(𝑘𝑒+𝑏𝑒⁢𝑠)⁢𝑋𝑚⁡(𝑠)
Submitting Eq. (A.7) into Eq. (A.4), one has the relationship between Xin(s) and ΔF(s):
(A.8)
Δ⁢𝐹⁡(𝑠)
𝑋𝑖⁢𝑛⁡(𝑠)
 
=−
𝑠2⁢(𝑚⁢𝑠+𝑏)⁢(𝑘𝑒+𝑏𝑒⁢𝑠)
𝐽+𝑘𝑖⁢𝑘𝑒
 
where J = (𝑚+𝑘𝑑⁢𝑏𝑒)⁢𝑠3+(𝑏+𝑘𝑑⁢𝑘𝑒+(1+𝑘𝑝)⁢𝑏𝑒)⁢𝑠2+(𝑘𝑒⁢(1+𝑘𝑝)+𝑘𝑖⁢𝑏𝑒)⁢𝑠.
Consider 𝑋𝑖⁢𝑛⁡(𝑠)=𝑎0/𝑠⥀, according to the Final value principle, one has:
(A.9)
lim
𝑠→0
 ⁡𝑠⁢Δ⁢𝐹⁡(𝑠)= 
lim
𝑠→0
 −
𝑠3⁢(𝑚⁢𝑠+𝑏)⁢(𝑘𝑒+𝑏𝑒⁢𝑠)
𝑘𝑖⁢𝑘𝑒
 
×
𝑎0
𝑠3
 
=−𝑎0⁢
𝑏
𝑘𝑖
 
This means that when the reference trajectory xin(t) possesses a constant acceleration, the force tracking error of the AVAC method maintains a steady-state error related to acceleration a0, virtual damping b, and the PID parameter ki.
Data availability
No data was used for the research described in the article.
References
Bao, Wang, Zheng, Housden, Hajnal, Rhode, 2023
X. Bao, S. Wang, L. Zheng, R.J. Housden, J.V. Hajnal, K. Rhode
A novel ultrasound robot with force/torque measurement and control for safe and efficient scanning
IEEE Transactions on Instrumentation and Measurement, 72 (2023), pp. 1-12
Google Scholar
Beber, Lamon, Nardi, Fontanelli, Saveriano, Palopoli, 2024
L. Beber, E. Lamon, D. Nardi, D. Fontanelli, M. Saveriano, L. Palopoli
A passive variable impedance control strategy with viscoelastic parameters estimation of soft tissues for safe ultrasonography
2024 IEEE International conference on robotics and automation (ICRA), IEEE (2024), pp. 1298-1304
CrossrefView in ScopusGoogle Scholar
Chen, Zhang, Zhao, Xie, Yang, Tao, Ding, 2024
Y. Chen, Y. Zhang, X. Zhao, Q. Xie, K. Yang, B. Tao, H. Ding
Physical human–robot interaction based on adaptive impedance control for robotic-assisted total hip arthroplasty
IEEE/ASME Transactions on Mechatronics, 29 (6) (2024), pp. 4674-4686
CrossrefView in ScopusGoogle Scholar
Duan, Gan, Chen, Dai, 2018
J. Duan, Y. Gan, M. Chen, X. Dai
Adaptive variable impedance control for dynamic contact force tracking in uncertain environment
Robotics and Autonomous systems, 102 (2018), pp. 54-65
View PDF
View articleView in ScopusGoogle Scholar
Ferraguti, Talignani Landi, Sabattini, Bonfe, Fantuzzi, Secchi, 2019
F. Ferraguti, C. Talignani Landi, L. Sabattini, M. Bonfe, C. Fantuzzi, C. Secchi
A variable admittance control strategy for stable physical human–robot interaction
The International Journal of Robotics Research, 38 (6) (2019), pp. 747-765
CrossrefView in ScopusGoogle Scholar
Fu, Burzo, Iovene, Zhao, Ferrigno, De Momi, 2024
J. Fu, I. Burzo, E. Iovene, J. Zhao, G. Ferrigno, E. De Momi
Optimization-based variable impedance control of robotic manipulator for medical contact tasks
IEEE Transactions on Instrumentation and Measurement, 73 (2024), pp. 1-8
Google Scholar
Fu, Maimone, Iovene, Zhao, Redaelli, Ferrigno, De Momi, 2025
J. Fu, G. Maimone, E. Iovene, J. Zhao, A. Redaelli, G. Ferrigno, E. De Momi
Human-inspired active compliant and passive shared control framework for robotic contact-rich tasks in medical applications
IEEE Transactions on Robotics, 41 (2025), pp. 2549-2568
CrossrefGoogle Scholar
Haddadi, Hashtrudi-Zaad, 2008
A. Haddadi, K. Hashtrudi-Zaad
Online contact impedance identification for robotic systems
2008 IEEE/RSJ International conference on intelligent robots and systems, IEEE (2008), pp. 974-980
View in ScopusGoogle Scholar
Hamedani, Sadeghian, Zekri, Sheikholeslam, Keshmiri, 2021
M.H. Hamedani, H. Sadeghian, M. Zekri, F. Sheikholeslam, M. Keshmiri
Intelligent impedance control using wavelet neural network for dynamic contact force tracking in unknown varying environments
Control Engineering Practice, 113 (2021), Article 104840
View PDF
View articleView in ScopusGoogle Scholar
Hogan, 1985
N. Hogan
Impedance control: An approach to manipulation: Part II-implementation
Journal of dynamic systems, measurement, and control, 107 (1) (1985), pp. 8-16
CrossrefGoogle Scholar
Huang, Gao, Wang, 2024
Q. Huang, B. Gao, M. Wang
Robot-assisted autonomous ultrasound imaging for carotid artery
IEEE Transactions on Instrumentation and Measurement, 73 (2024), pp. 1-9
Google Scholar
Jiang, Zhou, Bi, Zhou, Wendler, Navab, 2021
Z. Jiang, Y. Zhou, Y. Bi, M. Zhou, T. Wendler, N. Navab
Deformation-aware robotic 3d ultrasound
IEEE Robotics and Automation Letters, 6 (4) (2021), pp. 7675-7682
CrossrefView in ScopusGoogle Scholar
Jung, Hsia, Bonitz, 2004
S. Jung, T.C. Hsia, R.G. Bonitz
Force tracking impedance control of robot manipulators under unknown environment
IEEE Transactions on Control Systems Technology, 12 (3) (2004), pp. 474-483
View in ScopusGoogle Scholar
Koizumi, Warisawa, Nagoshi, Hashizume, Mitsuishi, 2009
N. Koizumi, S. Warisawa, M. Nagoshi, H. Hashizume, M. Mitsuishi
Construction methodology for a remote ultrasound diagnostic system
IEEE Transactions on robotics, 25 (3) (2009), pp. 522-538
View in ScopusGoogle Scholar
Mathiassen, Fjellin, Glette, Hol, Elle, 2016
K. Mathiassen, J.E. Fjellin, K. Glette, P.K. Hol, O.J. Elle
An ultrasound robotic system using the commercial robot UR5
Frontiers in Robotics and AI, 3 (2016), p. 1
View in ScopusGoogle Scholar
Munir, Al-Battal, Al-Sheghri, Becher, Noga, Punithakumar, 2025
K. Munir, A.F. Al-Battal, A. Al-Sheghri, H. Becher, M. Noga, K. Punithakumar
A survey of autonomous robotic ultrasound scanning systems
IEEE Access, 13 (2025), pp. 103178-103197
CrossrefView in ScopusGoogle Scholar
Ning, Chen, Zhang, Liao, 2021
G. Ning, J. Chen, X. Zhang, H. Liao
Force-guided autonomous robotic ultrasound scanning control method for soft uncertain environment
International Journal of Computer Assisted Radiology and Surgery, 16 (12) (2021), pp. 2189-2199
CrossrefView in ScopusGoogle Scholar
Niu, Yang, Zhang, Ji, Jiang, Liu, 2025
B. Niu, D. Yang, L. Zhang, Y. Ji, L. Jiang, H. Liu
Enhancing ultrasound scanning skills in a leader–follower robotic system through expert hand impedance regulation
IEEE Journal of Biomedical and Health Informatics, 29 (9) (2025), pp. 6678-6688
CrossrefView in ScopusGoogle Scholar
Roveda, Iannacci, Vicentini, Pedrocchi, Braghin, Tosatti, 2015
L. Roveda, N. Iannacci, F. Vicentini, N. Pedrocchi, F. Braghin, L.M. Tosatti
Optimal impedance force-tracking control design with impact formulation for interaction tasks
IEEE Robotics and Automation Letters, 1 (1) (2015), pp. 130-136
Google Scholar
Santos, Cortesão, 2013
L. Santos, R. Cortesão
Admittance control for robotic-assisted tele-echography
2013 16th International conference on advanced robotics (ICAR), IEEE (2013), pp. 1-7
Google Scholar
Seraji, 1994
H. Seraji
Adaptive admittance control: An approach to explicit force control in compliant motion
Proceedings of the 1994 IEEE international conference on robotics and automation, IEEE (1994), pp. 2705-2712
Google Scholar
Si, Wang, Yang, 2024
W. Si, N. Wang, C. Yang
Design and quantitative assessment of teleoperation-based human–robot collaboration method for robot-assisted sonography
IEEE Transactions on Automation Science and Engineering, 22 (2024), pp. 317-327
CrossrefView in ScopusGoogle Scholar
Wu, Feng, An, Carbone, Li, 2025
K. Wu, S. Feng, H. An, G. Carbone, W. Li
Evaluation of robot kinematic performance under motion constraints in a teleoperated robotic ultrasound system
Mechanism and Machine Theory, 207 (2025), Article 105952
View PDF
View articleView in ScopusGoogle Scholar
Xiao, Wang, Long, Yang, 2025
S. Xiao, T. Wang, Y. Long, L. Yang
Optimizing variable admittance control for remote ultrasound scanning under uncertain environment
IEEE Access, 13 (2025), pp. 83274-83284
CrossrefView in ScopusGoogle Scholar
Xie, Chong, Liu, Zhao, Wang, 2023
F. Xie, Z. Chong, X.J. Liu, H. Zhao, J. Wang
Precise and smooth contact force control for a hybrid mobile robot used in polishing
Robotics and Computer-Integrated Manufacturing, 83 (2023), Article 102573
View PDF
View articleView in ScopusGoogle Scholar
Yang, Hua, Ding, Li, 2024
Y. Yang, C. Hua, W. Ding, J. Li
Gesture recognition-based robust predefined-time admittance control of bimanual teleoperation without force/torque measurement
IEEE Transactions on Instrumentation and Measurement, 73 (2024), pp. 1-10
View PDF
View articleGoogle Scholar
Yoon, Na, Song, 2024
I. Yoon, M. Na, J.B. Song
Assembly of low-stiffness parts through admittance control with adaptive stiffness
Robotics and Computer-Integrated Manufacturing, 86 (2024), Article 102678
View PDF
View articleView in ScopusGoogle Scholar
Zhang, Yuan, Zou, 2022
T. Zhang, C. Yuan, Y. Zou
Online optimization method of controller parameters for robot constant force grinding based on deep reinforcement learning rainbow
Journal of Intelligent & Robotic Systems, 105 (4) (2022), p. 85
View PDF
View articleCrossrefGoogle Scholar
Zhang, Khamesee, 2017
X. Zhang, M.B. Khamesee
Adaptive force tracking control of a magnetically navigated microrobot in uncertain environment
IEEE/ASME Transactions on Mechatronics, 22 (4) (2017), pp. 1644-1651
View in ScopusGoogle Scholar