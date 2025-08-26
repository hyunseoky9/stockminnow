from flask import Flask, request, jsonify, redirect, url_for
from flask_cors import CORS
import os
import sys
import importlib.util
import numpy as np
import json
import datetime
import pickle

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
    # Serve the HTML file
    with open('src/html/index.html', 'r') as f:
        html_content = f.read()
    return html_content

@app.route('/simulation')
def simulation_page():
    print("=== SIMULATION PAGE REQUESTED ===")
    # Serve the simulation page
    with open('src/html/simulation.html', 'r') as f:
        html_content = f.read()
    return html_content

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
            'state': convert_to_serializable(current_state),
            'timestep': user_timesteps.get(username, 0),
            'max_timesteps': MAX_TIMESTEPS,
            'total_reward': user_episode_data.get(username, {}).get('total_reward', 0.0),
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
            # Parse the action string like "[0.1,0.1,0.1,0.7]" into a numpy array
            if action_string.startswith('[') and action_string.endswith(']'):
                # Remove brackets and split by comma
                action_list = json.loads(action_string)
            else:
                # Try to split by comma if no brackets
                action_list = [float(x.strip()) for x in action_string.split(',')]
            
            action = np.array(action_list)
            print(f"Parsed action: {action}")
            print(f"Action shape: {action.shape}")
            print(f"Action sum: {np.sum(action)}")
            
            # Validate that elements sum to approximately 1
            if not np.isclose(np.sum(action), 1.0, atol=1e-6):
                return jsonify({
                    'error': f'Action elements must sum to 1. Current sum: {np.sum(action):.6f}'
                }), 400
            
        except (ValueError, json.JSONDecodeError) as e:
            return jsonify({
                'error': f'Invalid action format. Expected format: "[0.1,0.1,0.1,0.7]". Error: {str(e)}'
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
            'state': convert_to_serializable(next_state),
            'reward': convert_to_serializable(reward),
            'total_reward': user_episode_data[username]['total_reward'],
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