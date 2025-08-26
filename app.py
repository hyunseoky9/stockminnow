from flask import Flask, request, jsonify, redirect, url_for, send_file, send_from_directory
from flask_cors import CORS
import os
import sys
import importlib.util
import numpy as np
import json
import datetime
import pickle
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def calculate_harmonic_mean(values):
    """Calculate harmonic mean of a list of values, excluding zeros"""
    if not values:
        return 0.0
    
    # Convert values to scalars if they are arrays/lists and filter out zeros and negative values
    scalar_values = []
    for v in values:
        if isinstance(v, (list, tuple)) and len(v) > 0:
            scalar_val = v[0]  # Take the first element if it's a list/array
        elif hasattr(v, 'item'):  # numpy scalar
            scalar_val = v.item()
        else:
            scalar_val = v
        
        if isinstance(scalar_val, (int, float)) and scalar_val > 0:
            scalar_values.append(scalar_val)
    
    if not scalar_values:
        return 0.0
    
    # Harmonic mean = n / (sum of 1/xi)
    return len(scalar_values) / sum(1/x for x in scalar_values)

def transform_state_for_display(state):
    """Transform raw state values to meaningful display values"""
    # First transform all values: np.exp(x) - 1
    transformed = [np.exp(x) - 1 for x in state]
    
    # Extract components
    age0_abundance = transformed[0:3]  # First 3: age 0 abundance for Angostura, Isleta, San Acacia
    age1plus_abundance = transformed[3:6]  # Next 3: age 1+ abundance
    springflow = transformed[6]  # 6th value: springflow in m^3
    wild_effective_pop = transformed[7]  # 7th value: wild effective population size

    # Calculate total abundance for each reach (age 0 + age 1+)
    total_abundances = [age0 + age1plus for age0, age1plus in zip(age0_abundance, age1plus_abundance)]
    percentage_age0 = [age0 / total * 100 if total > 0 else 0 for age0, total in zip(age0_abundance, total_abundances)]
    # Apply log10(x+1) transformation for display
    log_abundances = [np.log10(x + 1) for x in total_abundances]
    log_wild_effective_pop = np.log10(wild_effective_pop + 1)
    
    # Convert springflow from m^3 to KAF (kilo acre-feet)
    # 1 m^3 = 0.000810714 acre-feet, so divide by 1000 for kilo acre-feet
    kaf_springflow = springflow * 0.000810714 / 1000
    
    return {
        'log10_angostura_abundance': round(log_abundances[0], 3),
        'percentage_age0_angostura': round(percentage_age0[0], 1),
        'log10_isleta_abundance': round(log_abundances[1], 3),
        'percentage_age0_isleta': round(percentage_age0[1], 1),
        'log10_san_acacia_abundance': round(log_abundances[2], 3),
        'percentage_age0_san_acacia': round(percentage_age0[2], 1),
        'log10_wild_effective_population_size': round(log_wild_effective_pop, 3),
        'springflow_KAF': round(kaf_springflow, 3)
    }

def generate_state_plot(state, env):
    """Generate a bar plot visualization of the current state"""
    # Transform state to meaningful values
    state_data = transform_state_for_display(state)
    
    # Prepare data for plotting
    reaches = ["Angostura", "Isleta", "San Acacia"]
    reach_vals_log = [
        state_data['log10_angostura_abundance'],
        state_data['log10_isleta_abundance'],
        state_data['log10_san_acacia_abundance']
    ]
    percentages_age0 = [
        state_data['percentage_age0_angostura'],
        state_data['percentage_age0_isleta'],
        state_data['percentage_age0_san_acacia']
    ]
    
    wild_site = ["Wild Ne"]
    wild_val_log = [state_data['log10_wild_effective_population_size']]
    
    spring_site = ["Springflow"]
    spring_val = [state_data['springflow_KAF']]
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    
    # Get y-axis limits from environment
    abundance_ylim_max = np.log10(env.N0minmax[1]) + 2
    ne_ylim_max = np.log10(env.Neminmax[1])
    sflow_ylim_max = env.qminmax[1] * 0.000810714 / 1000

    # --- First subplot: 3 reaches ---
    colors_reaches = ["#1b9e77", "#33a02c", "#66c2a5"]
    bars1 = axes[0].bar(reaches, reach_vals_log, color=colors_reaches, edgecolor="black")
    for bar, val, pct in zip(bars1, reach_vals_log, percentages_age0):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + val*0.03,  # dynamic offset (3% of height)
                     f"{pct:.1f}% are\nage0", ha='center', va='bottom', fontsize=10)
    axes[0].set_title("Abundance by Reach", fontsize=20)
    axes[0].set_ylabel("log10(Abundance)", fontsize=15)
    axes[0].tick_params(axis='both', which='major', labelsize=12)
    axes[0].set_ylim(0, abundance_ylim_max)  # keeps it neat
    
    # --- Second subplot: Wild Ne ---
    axes[1].bar(wild_site, wild_val_log, color="#d95f02", edgecolor="black")
    axes[1].set_title("Wild Effective\n Population Size", fontsize=20)
    axes[1].set_ylabel("log10(Wild Ne)", fontsize=15)
    axes[1].tick_params(axis='both', which='major', labelsize=12)
    axes[1].set_ylim(0, ne_ylim_max)
    
    # --- Third subplot: Springflow (linear) ---
    axes[2].bar(spring_site, spring_val, color="skyblue", edgecolor="black")
    axes[2].set_title("Springflow", fontsize=20)
    axes[2].set_ylabel("Springflow at Otowi (KAF)", fontsize=15)
    axes[2].tick_params(axis='both', which='major', labelsize=12)
    axes[2].set_ylim(0, sflow_ylim_max)  # Dynamic y-limit based on actual value

    # Adjust layout
    plt.tight_layout()
    
    # Save plot to memory as base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
    img_buffer.seek(0)
    
    # Convert to base64 string
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    plt.close(fig)  # Close the figure to free memory
    img_buffer.close()
    
    return img_base64

def save_episode_data(username, episode_data, env):
    """Save episode data to a pickle file"""
    try:
        # Create data directory if it doesn't exist
        data_dir = 'data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Add end time to episode data
        episode_data['end_time'] = datetime.datetime.now().isoformat()
        
        # Create filename with new format: username_environmentname_date_time.pkl
        now = datetime.datetime.now()
        date_str = now.strftime("%Y%m%d")
        time_str = now.strftime("%H%M%S")
        env_name = env.envID.replace('.', '_')  # Replace dots with underscores for filename
        filename = f"{username}_{env_name}_{date_str}_{time_str}.pkl"
        filepath = os.path.join(data_dir, filename)
        
        # Save to pickle file
        with open(filepath, 'wb') as f:
            pickle.dump(episode_data, f)
        
        print(f"Episode data saved to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Error saving episode data: {str(e)}")
        return None

# Global dictionary to store active environments for each user
active_environments = {}
# Global dictionary to track timesteps for each user
user_timesteps = {}
# Global dictionary to store episode data for each user
user_episode_data = {}
# Maximum number of timesteps per episode
MAX_TIMESTEPS = 10

@app.route('/')
def index():
    print("=== INDEX PAGE REQUESTED ===")
    # Serve the HTML file with proper encoding
    with open('src/html/index.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content

@app.route('/simulation')
def simulation_page():
    print("=== SIMULATION PAGE REQUESTED ===")
    # Serve the simulation page with proper encoding
    with open('src/html/simulation.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content

@app.route('/images/<filename>')
def serve_image(filename):
    """Serve images from the src/images directory"""
    try:
        return send_from_directory('src/images', filename)
    except FileNotFoundError:
        return jsonify({'error': 'Image not found'}), 404

@app.route('/start-simulation', methods=['POST'])
def start_simulation():
    print("=== START SIMULATION CALLED ===")
    original_cwd = os.getcwd()  # Store this at the beginning
    
    try:
        data = request.get_json()
        print(f"Received data: {data}")
        username = data.get('username')
        print(f"Username: {username}")
        
        if not username:
            print("No username provided")
            return jsonify({'error': 'Username is required'}), 400
        
        # Add the environment directory to Python path so all dependencies can be found
        env_dir = os.path.abspath('src/env')
        print(f"Environment directory: {env_dir}")
        
        if env_dir not in sys.path:
            sys.path.insert(0, env_dir)
            print("Added env_dir to sys.path")
        
        # Change working directory to env directory so relative file paths work
        print(f"Changing working directory from {original_cwd} to {env_dir}")
        os.chdir(env_dir)
        
        try:
            print("Attempting to import Hatchery3_2_4...")
            # Import and initialize the environment
            spec = importlib.util.spec_from_file_location("hatchery_env", "Hatchery3_2_4.py")
            hatchery_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hatchery_module)
            print("Successfully imported module")
            
            print("Creating environment instance...")
            # Create environment instance (you may need to adjust these parameters)
            env = hatchery_module.Hatchery3_2_4(None, 1, -1, 1, 1)
            print("Environment created, calling reset...")
            
            initial_state = env.reset()
            print(f"Environment reset successful, initial state type: {type(initial_state)}")
            print(f'env initial state: {env.state}')
            # Store the environment for this user
            active_environments[username] = env
            # Initialize timestep counter
            user_timesteps[username] = 0
            
            # Convert initial state to serializable format
            def convert_to_serializable(obj):
                if hasattr(obj, 'tolist'):
                    return obj.tolist()
                elif hasattr(obj, 'item'):
                    return obj.item()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                else:
                    return str(obj)
            
            # Initialize episode data collection
            user_episode_data[username] = {
                'username': username,
                'start_time': datetime.datetime.now().isoformat(),
                'initial_state': convert_to_serializable(env.state),
                'parameter_version': getattr(env, 'paramsampleidx', None),
                'trajectory': [],  # Will store [state, action, reward, next_state, done] for each step
                'ne_scores': [],  # Will store Ne_score from each step
                'episode_length': 0,
                'total_reward': 0.0
            }
            
            print(f"Environment stored for user: {username}")
            print(f"Timestep counter initialized to 0")
            print(f"Episode data collection initialized")
            
            return jsonify({
                'success': True,
                'message': f'Simulation started for user: {username}',
                'initial_state': convert_to_serializable(initial_state),
                'max_timesteps': MAX_TIMESTEPS,
                'redirect': '/simulation'
            })
        
        finally:
            # Always restore the original working directory
            print(f"Restoring working directory to: {original_cwd}")
            os.chdir(original_cwd)
            
    except Exception as e:
        # Restore working directory in case of error too
        try:
            os.chdir(original_cwd)
        except:
            pass
        
        # Print detailed error information
        print(f"ERROR in start_simulation: {str(e)}")
        print(f"ERROR TYPE: {type(e).__name__}")
        import traceback
        print("FULL TRACEBACK:")
        traceback.print_exc()
        
        return jsonify({'error': f'Detailed error: {str(e)}'}), 500

@app.route('/get-current-state', methods=['POST'])
def get_current_state():
    try:
        data = request.get_json()
        username = data.get('username')
        
        if not username or username not in active_environments:
            return jsonify({'error': 'No active simulation for this user'}), 400
        
        env = active_environments[username]
        
        # Get current state from the environment
        current_state = env.state  # Assuming your env has a .state attribute
        
        # Generate the plot image as base64 string
        plot_image = generate_state_plot(current_state, env)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            else:
                return obj
        
        return jsonify({
            'success': True,
            'plot_image': plot_image,
            'timestep': user_timesteps.get(username, 0),
            'max_timesteps': MAX_TIMESTEPS,
            'total_reward': round(user_episode_data.get(username, {}).get('total_reward', 0.0), 2),
            'state_info': {
                'type': str(type(current_state)),
                'shape': getattr(current_state, 'shape', 'N/A') if hasattr(current_state, 'shape') else 'N/A'
            }
        })
            
    except Exception as e:
        print(f"ERROR in get_current_state: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/take-action', methods=['POST'])
def take_action():
    print("=== TAKE ACTION CALLED ===")
    
    try:
        data = request.get_json()
        username = data.get('username')
        action_string = data.get('action')
        
        print(f"Username: {username}")
        print(f"Action string: {action_string}")
        
        if not username or username not in active_environments:
            return jsonify({'error': 'No active simulation for this user'}), 400
        
        if not action_string:
            return jsonify({'error': 'Action is required'}), 400
        
        env = active_environments[username]
        
        try:
            # Parse the action string like "10,30,32,28" (percentages) into a numpy array
            if action_string.startswith('[') and action_string.endswith(']'):
                # Remove brackets if present and split by comma
                action_string = action_string.strip('[]')
                
            # Split by comma and convert percentages to decimals
            percentage_list = [float(x.strip()) for x in action_string.split(',')]
            
            # Convert percentages to decimals (divide by 100)
            action = np.array([p/100.0 for p in percentage_list])
            
            print(f"Parsed percentages: {percentage_list}")
            print(f"Converted to action: {action}")
            print(f"Action shape: {action.shape}")
            print(f"Action sum: {np.sum(action)}")
            
            # Validate that percentages sum to approximately 100 (action sum to 1)
            if not np.isclose(np.sum(percentage_list), 100.0, atol=1e-6):
                return jsonify({
                    'error': f'Percentages must sum to 100. Current sum: {np.sum(percentage_list):.1f}%'
                }), 400
            
        except (ValueError, json.JSONDecodeError) as e:
            return jsonify({
                'error': f'Invalid percentage format. Expected format: "10,30,32,28". Error: {str(e)}'
            }), 400
        
        # Take the action in the environment
        print(f"Taking action: {action}")
        print(f"state: {env.state}")
        current_state = env.state.copy() if hasattr(env.state, 'copy') else env.state  # Store state before step
        next_state, reward, done, info = env.step(action)
        
        # Increment timestep counter
        user_timesteps[username] += 1
        current_timestep = user_timesteps[username]
        
        print(f"next state: {next_state}")
        print(f"reward: {reward}")
        print(f"Step result - Timestep: {current_timestep}/{MAX_TIMESTEPS}, Reward: {reward}, Done: {done}")
        print(f"New state type: {type(next_state)}")
        
        # Check if we've reached the maximum timesteps
        if current_timestep >= MAX_TIMESTEPS:
            done = True
            print(f"Maximum timesteps ({MAX_TIMESTEPS}) reached for user: {username}")
        
        # Collect episode data
        def convert_to_serializable(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            else:
                return obj
        
        # Add this step to the trajectory
        step_data = {
            'timestep': current_timestep,
            'state': convert_to_serializable(current_state),
            'action': action.tolist(),
            'reward': convert_to_serializable(reward),
            'next_state': convert_to_serializable(next_state),
            'done': bool(done),
            'info': {k: convert_to_serializable(v) for k, v in (info or {}).items()}
        }
        
        user_episode_data[username]['trajectory'].append(step_data)
        user_episode_data[username]['episode_length'] = current_timestep
        user_episode_data[username]['total_reward'] += convert_to_serializable(reward)
        
        # Collect Ne_score if available in info
        if info and 'Ne_score' in info:
            ne_score_raw = info['Ne_score']
            ne_score = convert_to_serializable(ne_score_raw)
            
            # Ensure we store a scalar value, not a list
            if isinstance(ne_score, (list, tuple)) and len(ne_score) > 0:
                ne_score = ne_score[0]
            elif hasattr(ne_score, 'item'):
                ne_score = ne_score.item()
                
            user_episode_data[username]['ne_scores'].append(ne_score)
            print(f"Ne_score collected: {ne_score} (original type: {type(ne_score_raw)})")
        
        print(f"Step data collected for timestep {current_timestep}")
        print(f"Reward being sent to frontend: {convert_to_serializable(reward)}")
        
        # Convert info dict if it contains numpy arrays
        serializable_info = {}
        if info:
            for key, value in info.items():
                serializable_info[key] = convert_to_serializable(value)
        
        response = {
            'success': True,
            'plot_image': generate_state_plot(next_state, env),
            'reward': round(convert_to_serializable(reward), 2),
            'total_reward': round(user_episode_data[username]['total_reward'], 2),
            'done': bool(done),
            'info': serializable_info,
            'action_taken': action.tolist(),
            'timestep': current_timestep,
            'max_timesteps': MAX_TIMESTEPS
        }
        
        # If episode is done, clean up
        if done:
            # Calculate long-term Ne (harmonic mean of Ne_scores)
            ne_scores = user_episode_data[username]['ne_scores']
            long_term_ne = calculate_harmonic_mean(ne_scores)
            user_episode_data[username]['long-term Ne'] = long_term_ne
            
            total_reward = user_episode_data[username]['total_reward']
            
            # Save episode data before cleanup
            print(f"Episode completed! Saving data for user: {username}")
            saved_file = save_episode_data(username, user_episode_data[username], env)
            if saved_file:
                response['saved_file'] = saved_file
                print(f"Episode data saved successfully to: {saved_file}")
            else:
                print("Failed to save episode data")
            
            # Cleanup
            del active_environments[username]
            del user_timesteps[username]
            del user_episode_data[username]  # Clean up episode data after saving
            
            # Create custom completion message with total reward and long-term Ne
            completion_message = f'Episode completed! Total reward: {total_reward:.2f}, Long-term Ne: {long_term_ne:.2f}'
            response['message'] = completion_message
            print(f"Episode completed for user: {username} - {completion_message}")
        
        return jsonify(response)
            
    except Exception as e:
        print(f"ERROR in take_action: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/reset-episode', methods=['POST'])
def reset_episode():
    print("=== RESET EPISODE CALLED ===")
    original_cwd = os.getcwd()
    
    try:
        data = request.get_json()
        username = data.get('username')
        
        print(f"Username: {username}")
        
        if not username:
            return jsonify({'error': 'Username is required'}), 400
        
        # Clean up any existing environment for this user
        if username in active_environments:
            del active_environments[username]
        if username in user_timesteps:
            del user_timesteps[username]
        if username in user_episode_data:
            del user_episode_data[username]
        
        # Add the environment directory to Python path so all dependencies can be found
        env_dir = os.path.abspath('src/env')
        print(f"Environment directory: {env_dir}")
        
        if env_dir not in sys.path:
            sys.path.insert(0, env_dir)
            print("Added env_dir to sys.path")
        
        # Change working directory to env directory so relative file paths work
        print(f"Changing working directory from {original_cwd} to {env_dir}")
        os.chdir(env_dir)
        
        try:
            print("Attempting to import Hatchery3_2_4...")
            # Import and initialize the environment
            spec = importlib.util.spec_from_file_location("hatchery_env", "Hatchery3_2_4.py")
            hatchery_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hatchery_module)
            print("Successfully imported module")
            
            print("Creating environment instance...")
            # Create environment instance (you may need to adjust these parameters)
            env = hatchery_module.Hatchery3_2_4(None, 1, -1, 1, 1)
            print("Environment created, calling reset...")
            
            initial_state = env.reset()
            print(f"Environment reset successful, initial state type: {type(initial_state)}")
            print(f'env initial state: {env.state}')
            
            # Store the environment for this user
            active_environments[username] = env
            # Initialize timestep counter
            user_timesteps[username] = 0
            print(f"Environment stored for user: {username}")
            print(f"Timestep counter initialized to 0")
            
            # Convert current state to serializable format
            def convert_to_serializable(obj):
                if hasattr(obj, 'tolist'):
                    return obj.tolist()
                elif hasattr(obj, 'item'):
                    return obj.item()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                else:
                    return str(obj)
            
            # Use env.state instead of initial_state for consistency with other endpoints
            current_state = env.state
            
            # Initialize episode data collection
            user_episode_data[username] = {
                'username': username,
                'start_time': datetime.datetime.now().isoformat(),
                'initial_state': convert_to_serializable(current_state),
                'parameter_version': getattr(env, 'paramsampleidx', None),
                'trajectory': [],  # Will store [state, action, reward, next_state, done] for each step
                'ne_scores': [],  # Will store Ne_score from each step
                'episode_length': 0,
                'total_reward': 0.0
            }
            print(f"Episode data collection initialized")
            
            return jsonify({
                'success': True,
                'message': f'Episode reset for user: {username}',
                'initial_state': convert_to_serializable(current_state),
                'timestep': 0,
                'max_timesteps': MAX_TIMESTEPS
            })
        
        finally:
            # Always restore the original working directory
            print(f"Restoring working directory to: {original_cwd}")
            os.chdir(original_cwd)
            
    except Exception as e:
        # Restore working directory in case of error too
        try:
            os.chdir(original_cwd)
        except:
            pass
        
        # Print detailed error information
        print(f"ERROR in reset_episode: {str(e)}")
        print(f"ERROR TYPE: {type(e).__name__}")
        import traceback
        print("FULL TRACEBACK:")
        traceback.print_exc()
        
        return jsonify({'error': f'Detailed error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)