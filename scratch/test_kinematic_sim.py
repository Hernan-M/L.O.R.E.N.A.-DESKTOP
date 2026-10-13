import numpy as np

def simulate_kinematic_forwarding():
    # Simulate a saccade movement from 200px to 800px over 10 frames (300ms)
    t = 0.0
    dt = 0.033 # 30 FPS
    
    positions = [200.0, 205.0, 210.0, 280.0, 420.0, 600.0, 750.0, 795.0, 800.0, 800.0]
    
    lead_time_s = 0.020
    v_thresh = 80.0
    v_full = 300.0
    
    p_prev = None
    t_prev = None
    v_filt = np.zeros(2)
    
    print("Frame | Raw Pos | Speed (px/s) | Weight | Pred Pos | Lead (px)")
    print("-" * 65)
    for i, p_val in enumerate(positions):
        p = np.array([p_val, 500.0])
        t += dt
        
        if p_prev is not None and t_prev is not None:
            inst_v = (p - p_prev) / (t - t_prev)
            v_filt = v_filt + 0.6 * (inst_v - v_filt)
            speed = float(np.linalg.norm(v_filt))
            weight = float(np.clip((speed - v_thresh) / (v_full - v_thresh), 0.0, 1.0))
            p_forward = p + v_filt * (lead_time_s * weight)
        else:
            speed = 0.0
            weight = 0.0
            p_forward = p.copy()
            
        p_prev = p.copy()
        t_prev = t
        
        lead_px = p_forward[0] - p[0]
        print(f"{i:5d} | {p[0]:7.1f} | {speed:12.1f} | {weight:6.2f} | {p_forward[0]:8.1f} | {lead_px:+8.1f}")

if __name__ == "__main__":
    simulate_kinematic_forwarding()
