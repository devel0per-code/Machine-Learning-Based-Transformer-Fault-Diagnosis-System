import pandas as pd
import numpy as np

def generate_synthetic_dga(num_samples=1000):
    np.random.seed(42)
    data = []
    
    # Fault types and their rough gas distributions
    # Gases: H2, CH4, C2H4, C2H2, CO, CO2
    fault_types = {
        'Normal': {'H2': (10, 50), 'CH4': (10, 30), 'C2H4': (1, 10), 'C2H2': (0, 1), 'CO': (100, 300), 'CO2': (1000, 3000)},
        'PD': {'H2': (100, 500), 'CH4': (50, 150), 'C2H4': (1, 10), 'C2H2': (0, 1), 'CO': (100, 300), 'CO2': (1000, 3000)},
        'T1': {'H2': (10, 50), 'CH4': (50, 200), 'C2H4': (10, 50), 'C2H2': (0, 2), 'CO': (100, 500), 'CO2': (1000, 4000)},
        'T2': {'H2': (10, 50), 'CH4': (50, 200), 'C2H4': (50, 150), 'C2H2': (0, 2), 'CO': (100, 500), 'CO2': (1000, 4000)},
        'T3': {'H2': (10, 50), 'CH4': (50, 200), 'C2H4': (150, 500), 'C2H2': (0, 5), 'CO': (100, 500), 'CO2': (1000, 4000)},
        'D1': {'H2': (200, 800), 'CH4': (50, 200), 'C2H4': (10, 50), 'C2H2': (50, 200), 'CO': (100, 500), 'CO2': (1000, 4000)},
        'D2': {'H2': (200, 800), 'CH4': (50, 200), 'C2H4': (50, 200), 'C2H2': (100, 500), 'CO': (100, 500), 'CO2': (1000, 4000)},
    }
    
    for fault, ranges in fault_types.items():
        # Distribute samples evenly, but more Normal
        n = num_samples // len(fault_types)
        if fault == 'Normal':
            n = num_samples - (len(fault_types) - 1) * n
        
        for _ in range(n):
            h2 = np.random.uniform(*ranges['H2'])
            ch4 = np.random.uniform(*ranges['CH4'])
            c2h4 = np.random.uniform(*ranges['C2H4'])
            c2h2 = np.random.uniform(*ranges['C2H2'])
            co = np.random.uniform(*ranges['CO'])
            co2 = np.random.uniform(*ranges['CO2'])
            
            # Add some noise
            h2 += np.random.normal(0, h2*0.1)
            ch4 += np.random.normal(0, ch4*0.1)
            c2h4 += np.random.normal(0, c2h4*0.1)
            c2h2 += np.random.normal(0, c2h2*0.1)
            co += np.random.normal(0, co*0.1)
            co2 += np.random.normal(0, co2*0.1)
            
            data.append([max(0, h2), max(0, ch4), max(0, c2h4), max(0, c2h2), max(0, co), max(0, co2), fault])
            
    df = pd.DataFrame(data, columns=['H2', 'CH4', 'C2H4', 'C2H2', 'CO', 'CO2', 'Fault_Type'])
    return df

if __name__ == "__main__":
    df = generate_synthetic_dga(2000)
    df.to_csv('synthetic_dga_data.csv', index=False)
    print("Generated synthetic_dga_data.csv")
