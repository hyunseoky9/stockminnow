from flask import Flask, request, jsonify, redirect, url_for, send_file, send_from_directory
from flask_cors import CORS
import os
import sys
import importlib.util
import numpy as np
import json
import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def index():
    print("=== INDEX PAGE REQUESTED ===")
    # Serve the HTML file with proper encoding
    with open('src/html/index.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content

@app.route('/decision-tool')
def decision_tool_page():
    print("=== DECISION TOOL PAGE REQUESTED ===")
    # Serve the decision tool page with proper encoding
    with open('src/html/decision-tool.html', 'r', encoding='utf-8') as f:
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
        print(f"ne_score:{info['Ne_score']}")
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

@app.route('/get-rl-recommendations', methods=['POST'])
def get_rl_recommendations():
    print("=== GET RL RECOMMENDATIONS CALLED ===", flush=True)
    import sys
    sys.stdout.flush()
    original_cwd = os.getcwd()
    
    try:
        import torch
        import torch.nn as nn
        
        data = request.get_json()
        print(f"Received input data: {data}", flush=True)
        
        # Validate input data
        required_fields = ['age0_angostura', 'age0_isleta', 'age0_san_acacia', 
                          'age1plus_angostura', 'age1plus_isleta', 'age1plus_san_acacia',
                          'july1_age0_angostura', 'july1_age0_isleta', 'july1_age0_san_acacia',
                          'spring_age0_angostura', 'spring_age0_isleta', 'spring_age0_san_acacia',
                          'spring_age1plus_angostura', 'spring_age1plus_isleta', 'spring_age1plus_san_acacia',
                          'spring_flow']
        
        print(f"Validating {len(required_fields)} required fields...", flush=True)
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Add the environment directory to Python path
        env_dir = os.path.abspath('src/env')
        if env_dir not in sys.path:
            sys.path.insert(0, env_dir)
        
        # Change working directory to env directory
        os.chdir(env_dir)
        
        try:
            # Import and create environment
            spec = importlib.util.spec_from_file_location("hatchery_env", "Hatchery3_2_4.py")
            hatchery_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hatchery_module)
            
            # Create environment instance
            env = hatchery_module.Hatchery3_2_4(None, 1, -1, 1, 1)
            
            # Process fall abundance estimates with log(x+1) transformation
            fall_abundances = [
                np.log(data['age0_angostura'] + 1),
                np.log(data['age0_isleta'] + 1), 
                np.log(data['age0_san_acacia'] + 1),
                np.log(data['age1plus_angostura'] + 1),
                np.log(data['age1plus_isleta'] + 1),
                np.log(data['age1plus_san_acacia'] + 1)
            ]
            
            # Process spring populations for Ne calculation
            spN0 = np.array([
                data['spring_age0_angostura'],
                data['spring_age0_isleta'], 
                data['spring_age0_san_acacia']
            ])
            
            spN1 = np.array([
                data['spring_age1plus_angostura'],
                data['spring_age1plus_isleta'],
                data['spring_age1plus_san_acacia']
            ])
            
            # Calculate effective spawner population and generation time
            P = np.array([data['july1_age0_angostura'],
                          data['july1_age0_isleta'], 
                          data['july1_age0_san_acacia']
            ])
            effspawner = np.sum(spN0) + np.sum(spN1) * env.beta_2
            Denom = env.alpha*effspawner / P

            P1 = (env.alpha * spN0) / Denom
            P2 = (env.alpha * env.beta_2 * spN1) / Denom
            
            # Calculate generation time
            genT = (np.sum(P1) + np.sum(P2) * env.AVGage_of_age2plus) / np.sum(P)
            
            # Calculate wild Ne using env.NeCalc0
            Ne_wild, _, _ = env.NeCalc0(spN0, spN1, None, None, genT, None, 0)
            
            # Ensure Ne_wild is a scalar (extract from array if needed)
            if isinstance(Ne_wild, np.ndarray):
                Ne_wild = float(Ne_wild.item()) if Ne_wild.size == 1 else float(Ne_wild[0])
            
            # Process hydrology: convert KAF to cubic meters and apply log(x+1)
            spring_flow_m3 = data['spring_flow'] * 1233.48  # 1 KAF = 1233.48 m^3
            log_hydrology = np.log(spring_flow_m3 + 1)
            
            # Create state vector for RL model
            # [fall_abundances (6), log(Ne_wild), log_hydrology]
            state_vector = fall_abundances + [float(np.log(Ne_wild + 1)), float(log_hydrology)]
            
            # Convert to PyTorch tensor
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)  # Add batch dimension
            
            print(f"State vector shape: {state_tensor.shape}")
            print(f"State vector: {state_vector}")
            
            # Load RL model using the ddpg_actor module
            model_path = os.path.join(original_cwd, 'src', 'RL_model', 'RL_model.pt')
            print(f"Loading model from: {model_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"RL model not found at {model_path}")
            
            # Add RL_model directory to Python path for ddpg_actor module
            rl_model_dir = os.path.join(original_cwd, 'src', 'RL_model')
            if rl_model_dir not in sys.path:
                sys.path.insert(0, rl_model_dir)
            
            device = torch.device('cpu')
            model = torch.load(model_path, map_location=device, weights_only=False)
            model.eval()
            
            print(f"Model loaded successfully")
            print(f"Model type: {type(model)}")
            
            # Get prediction from RL model
            with torch.no_grad():
                action_probs = model(state_tensor)
                print(f"Raw model output: {action_probs}")
                print(f"Output shape: {action_probs.shape}")
                
                # Handle different output formats
                if len(action_probs.shape) > 1:
                    action_probs = action_probs.squeeze()
                
                # Convert to numpy if it's still a tensor
                if hasattr(action_probs, 'numpy'):
                    percentages = action_probs.numpy()
                else:
                    percentages = np.array(action_probs)
                
                # Ensure we have 4 values
                if len(percentages) != 4:
                    raise ValueError(f"Expected 4 action values, got {len(percentages)}")
                
                # Convert to percentages and normalize to sum to 100
                percentages = percentages * 100.0
                total = np.sum(percentages)
                if total > 0:
                    percentages = percentages * (100.0 / total)
                else:
                    # Fallback if all outputs are zero
                    percentages = np.array([25.0, 25.0, 25.0, 25.0])
                
                recommendations = {
                    'angostura': round(float(percentages[0]), 1),
                    'isleta': round(float(percentages[1]), 1),
                    'san_acacia': round(float(percentages[2]), 1),
                    'unstocked': round(float(percentages[3]), 1)
                }
            
            print(f"Final recommendations: {recommendations}")
            
            return jsonify({
                'success': True,
                'recommendations': recommendations
            })
            
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
            
    except Exception as e:
        # Restore working directory in case of error
        try:
            os.chdir(original_cwd)
        except:
            pass
        
        print(f"ERROR in get_rl_recommendations: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({'error': f'Error generating recommendations: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)