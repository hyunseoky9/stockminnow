from flask import Flask, request, jsonify, redirect, url_for, send_file, send_from_directory
from flask_cors import CORS
from functools import lru_cache
from pathlib import Path
import os
import sys
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import torch
matplotlib.use('Agg')  # Use non-interactive backend
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import json
import datetime
import pickle
import base64
import io

# Fix OpenMP duplicate library issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Base paths and RL model configuration
BASE_DIR = Path(__file__).resolve().parent
MODEL_SEED = 406580
MODEL_PARAMSET = 138

RL_MODEL_DIR = BASE_DIR / 'src' / 'RL_model'
if str(RL_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(RL_MODEL_DIR))

MODEL_PATH = RL_MODEL_DIR / f'seed{MODEL_SEED}_paramset{MODEL_PARAMSET}' / 'bestPolicyNetwork_Hatchery3.3.7_par0_dis-1_TD3.pt'
RMS_PATH = RL_MODEL_DIR / f'seed{MODEL_SEED}_paramset{MODEL_PARAMSET}' / 'rms_Hatchery3.3.7_par0_dis-1_TD3.pkl'


@lru_cache(maxsize=1)
def get_model_and_rms():
    """Load and cache the TD3 model and RMS normalizer once per process.

    This avoids repeated heavy imports and unpickling on every request,
    which is critical for constrained environments like Render Free.
    """
    import pickle

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"RL model not found at {MODEL_PATH}")
    if not RMS_PATH.exists():
        raise FileNotFoundError(f"Standardization file not found at {RMS_PATH}")

    print(f"Loading RL model from: {MODEL_PATH}")
    model = torch.load(str(MODEL_PATH), map_location='cpu', weights_only=False)
    model.eval()

    print(f"Loading RMS normalization data from: {RMS_PATH}")
    with open(RMS_PATH, 'rb') as f:
        rms = pickle.load(f)

    print("Model and RMS loaded and cached successfully")
    return model, rms


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

@app.route('/img/<filename>')
def serve_img(filename):
    """Serve favicon or other assets from the top-level img directory"""
    try:
        return send_from_directory('img', filename)
    except FileNotFoundError:
        return jsonify({'error': 'Image not found'}), 404

@app.route('/get-rl-recommendations', methods=['POST'])
def get_rl_recommendations():
    print("=== GET RL RECOMMENDATIONS CALLED ===", flush=True)
    import sys
    sys.stdout.flush()
    original_cwd = os.getcwd()
    
    try:
        data = request.get_json()
        print(f"Received input data: {data}", flush=True)
        
        # Check decision type
        decision_type = data.get('decision_type', 'production')  # Default to production if not specified
        print(f"Decision type: {decision_type}", flush=True)
        
        # Define required fields based on decision type
        if decision_type == 'production':
            required_fields = [
                'prod-total-catch-AON-angostura','prod-total-catch-AON-isleta','prod-total-catch-AON-san-acacia',
                'prod-total-catch-JAS-angostura','prod-total-catch-JAS-isleta','prod-total-catch-JAS-san-acacia',
                'prod-spring-flow'
            ]
        elif decision_type == 'distribution':
            required_fields = [
                'dist-total-catch-AON-angostura','dist-total-catch-AON-isleta','dist-total-catch-AON-san-acacia',
                'dist-total-catch-JAS-angostura','dist-total-catch-JAS-isleta','dist-total-catch-JAS-san-acacia',
                'dist-spring-flow','dist-hatchery-production'
            ]
        else:
            return jsonify({'error': f'Invalid decision_type: {decision_type}. Must be "production" or "distribution"'}), 400
        
        print(f"Validating {len(required_fields)} required fields...", flush=True)
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        try:

            if decision_type == 'production':
                logaoncatch = [
                    np.log(data['prod-total-catch-AON-angostura'] + 1),
                    np.log(data['prod-total-catch-AON-isleta'] + 1),
                    np.log(data['prod-total-catch-AON-san-acacia'] + 1),
                ]
                logjascatch = [
                    np.log(data['prod-total-catch-JAS-angostura'] + 1),
                    np.log(data['prod-total-catch-JAS-isleta'] + 1),
                    np.log(data['prod-total-catch-JAS-san-acacia'] + 1),
                ]
                springflow = np.log(data['prod-spring-flow']*1233481.84 + 1)  # Convert KAF to cubic meters
                nh = 0 # No hatchery production for production decision
                t = 0  # spring
                logtotalaoncatch = np.log10(sum([
                    data['prod-total-catch-AON-angostura'],
                    data['prod-total-catch-AON-isleta'],
                    data['prod-total-catch-AON-san-acacia'],
                ]) + 1)
                loglocalminaoncatch = np.log10(min([
                    data['prod-total-catch-AON-angostura'],
                    data['prod-total-catch-AON-isleta'],
                    data['prod-total-catch-AON-san-acacia'],
                ]) + 1)
                springflowkaf = data['prod-spring-flow']
            elif decision_type == 'distribution':
                logaoncatch = [
                    np.log(data['dist-total-catch-AON-angostura'] + 1),
                    np.log(data['dist-total-catch-AON-isleta'] + 1),
                    np.log(data['dist-total-catch-AON-san-acacia'] + 1),
                ]
                logjascatch = [
                    np.log(data['dist-total-catch-JAS-angostura'] + 1),
                    np.log(data['dist-total-catch-JAS-isleta'] + 1),
                    np.log(data['dist-total-catch-JAS-san-acacia'] + 1),
                ]
                springflow = np.log(data['dist-spring-flow']*1233481.84 + 1)  # Convert KAF to cubic meters
                nh = np.log(data['dist-hatchery-production'] + 1)
                t = 1 # fall
            

            # Create state vector for RL model
            # [fall_abundances (6), log(Ne_wild), log_hydrology]
            state_vector = np.array(logaoncatch + logjascatch + [nh, springflow, t])
            
            
            # Load TD3 RL model and RMS normalizer (cached at process level)
            model, rms = get_model_and_rms()
            print(f"Model type: {type(model)}")

            # Standardize state vector
            state_vector_std = rms.normalize(state_vector)
            print('standardization complete')
            # Convert to PyTorch tensor
            state_tensor = torch.FloatTensor(state_vector_std).unsqueeze(0)  # Add batch dimension
            
            print(f"State vector shape: {state_tensor.shape}")
            print(f"Original state vector: {state_vector}")
            print(f"Standardized state vector: {state_tensor}")
            
            
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
                
                 
                prod_recomm = percentages[0]
                dist_recomm = percentages[1:]/np.sum(percentages[1:])

                if decision_type == 'production':
                    recommendations = {
                        'production_recommendation': float(np.round(prod_recomm, 3)*100)
                    }
                elif decision_type == 'distribution':
                    
                    recommendations = {
                        'angostura': float(np.round(dist_recomm[0], 3)*100),
                        'isleta': float(np.round(dist_recomm[1], 3)*100),
                        'san_acacia': float(np.round(dist_recomm[2], 3)*100),
                    }
                    recommendeddist = np.array([dist_recomm[0], dist_recomm[1], dist_recomm[2]])
            print(f"Final {decision_type} recommendations: {recommendations}")

            # for production support, get the production action transition file.
            if decision_type == 'production':
                prodaction_path = os.path.join(
                    original_cwd,
                    'src',
                    'RL_model',
                    f'seed{MODEL_SEED}_paramset{MODEL_PARAMSET}',
                    f'simulation_spring_transitions_seed{MODEL_SEED}_paramset{MODEL_PARAMSET}_c7_Hatchery3.3.7.csv'
                )
                
                # Load the CSV file as pandas DataFrame
                try:
                    springdf = pd.read_csv(prodaction_path)
                    print(f"Loaded production action CSV with shape: {springdf.shape}")
                    print(f"Columns: {springdf.columns.tolist()}")
                    
                    # Create scatter plots for interpretation
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                    
                    # First plot: Production vs Flow and Total Catch
                    sc1 = ax1.scatter(springdf['q_kaf'], 
                                     np.log10(springdf['catch_total_apr+oct+nov']+1), 
                                     c=springdf['production'], 
                                     cmap='viridis', 
                                     s=5, 
                                     vmin=0, 
                                     vmax=1)
                    cbar1 = plt.colorbar(sc1, ax=ax1, label='Production Level')
                    ax1.set_xlabel('Forecasted Spring Flow (KAF)', fontsize=16)
                    ax1.set_ylabel('Log total catch in \nApr and last year Oct + Nov', fontsize=16)
                    ax1.set_title('Production Decisions on\n Flow and Total Catch space', fontsize=18)
                    ax1.scatter(springflowkaf, logtotalaoncatch, c='red', s=50, label='Current Data Point')
                    ax1.legend()
                    
                    # Second plot: Production vs Total Catch and Min Catch
                    sc2 = ax2.scatter(np.log10(springdf['catch_total_apr+oct+nov']+1), 
                                     np.log10(springdf['min_catch_apr+oct+nov_reach']+1), 
                                     c=springdf['production'], 
                                     cmap='viridis', 
                                     s=5,
                                     vmin=0, 
                                     vmax=1)
                    cbar2 = plt.colorbar(sc2, ax=ax2, label='Production level')
                    cbar2.ax.tick_params(labelsize=14)
                    cbar2.set_label('Production level', fontsize=14)
                    ax2.set_xlabel('Log total catch in \nApr and last year Oct + Nov', fontsize=16)
                    ax2.set_ylabel('Minimum log local catch in \nApr and last year Oct + Nov', fontsize=16)
                    ax2.set_title('Production Decisions on\n Total Catch and Min Local Catch space', fontsize=18)
                    ax2.scatter(logtotalaoncatch, loglocalminaoncatch, c='red', s=50, label='Current Data Point')
                    ax2.legend()
                    ax2.tick_params(axis='both', which='major', labelsize=14)
                    plt.tight_layout()

                    # Save first plot to base64 string
                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                    img_buffer.seek(0)
                    plot_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                    plt.close()
                    
                    # Plot the histogram of Apr-Oct-Nov catches from simulation and mark the input catch in it for each reach 
                    # Also plot histogram of spring flow with current input marked
                    fig2, axs2 = plt.subplots(2, 2, figsize=(16, 10))
                    axs2 = axs2.flatten()  # Flatten for easier indexing
                    
                    reaches = ['angostura', 'isleta', 'san-acacia']
                    reachnames = ['Angostura', 'Isleta', 'San Acacia']
                    reachidx = ['a','i','s']
                    for i, reach in enumerate(reaches):
                        axs2[i].hist(np.log10(springdf[f'catch_apr+oct+nov_{reachidx[i]}']+1), bins=30, color='skyblue', edgecolor='black')
                        axs2[i].axvline(x=(np.log10(data[f'prod-total-catch-AON-{reach}']+1)), color='red', linestyle='--', label='Current Input Catch', linewidth=2)
                        axs2[i].set_title(f'Catch Distribution for {reachnames[i]}', fontsize=16)
                        axs2[i].set_xlabel('Log Catch (Apr + Oct + Nov)', fontsize=14)
                        axs2[i].set_ylabel('Frequency', fontsize=14)
                        axs2[i].legend()
                    
                    # Add spring flow histogram as the 4th subplot
                    axs2[3].hist(springdf['q_kaf'], bins=30, color='lightgreen', edgecolor='black')
                    axs2[3].axvline(x=springflowkaf, color='red', linestyle='--', label='Current Input Flow', linewidth=2)
                    axs2[3].set_title('Spring Flow Distribution', fontsize=16)
                    axs2[3].set_xlabel('Forecasted Spring Flow (KAF)', fontsize=14)
                    axs2[3].set_ylabel('Frequency', fontsize=14)
                    axs2[3].legend()
                    
                    plt.tight_layout()



                    # Save second plot to base64 string
                    img_buffer2 = io.BytesIO()
                    plt.savefig(img_buffer2, format='png', dpi=150, bbox_inches='tight')
                    img_buffer2.seek(0)
                    histogram_base64 = base64.b64encode(img_buffer2.getvalue()).decode('utf-8')
                    plt.close()
                    
                    # Add both plots to recommendations
                    recommendations['plot_data'] = f"data:image/png;base64,{plot_base64}"
                    recommendations['histogram_data'] = f"data:image/png;base64,{histogram_base64}"
                    
                except FileNotFoundError:
                    print(f"WARNING: Production action CSV not found at {prodaction_path}")
                    springdf = None
                except Exception as e:
                    print(f"ERROR loading production CSV or creating plot: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    springdf = None
                except Exception as e:
                    print(f"ERROR loading production CSV: {str(e)}")
                    springdf = None
            elif decision_type == 'distribution':
                distaction_path = os.path.join(
                    original_cwd,
                    'src',
                    'RL_model',
                    f'seed{MODEL_SEED}_paramset{MODEL_PARAMSET}',
                    f'simulation_fall_transitions_seed{MODEL_SEED}_paramset{MODEL_PARAMSET}_c7_Hatchery3.3.7.csv'
                )
                try:
                    falldf_analysis = pd.read_csv(distaction_path)
                    print(f"Loaded distribution action CSV with shape: {falldf_analysis.shape}")
                    print(f"Columns: {falldf_analysis.columns.tolist()}")
                    # Debug: Check the data
                    stock_a = falldf_analysis['stock_a']
                    stock_i = falldf_analysis['stock_i']
                    stock_s = falldf_analysis['stock_s']
                    print(f"Stock data ranges - a: {stock_a.min():.3f} to {stock_a.max():.3f}")
                    print(f"Stock data ranges - i: {stock_i.min():.3f} to {stock_i.max():.3f}") 
                    print(f"Stock data ranges - s: {stock_s.min():.3f} to {stock_s.max():.3f}")
                    print(f"Number of data points: {len(falldf_analysis)}")
                    
                    # Check if data sums to 1 (required for ternary plots)
                    stock_sums = stock_a + stock_i + stock_s
                    print(f"Stock sum ranges: {stock_sums.min():.3f} to {stock_sums.max():.3f}")
                    print(f"Stock sum mean: {stock_sums.mean():.3f}")
                    valid_mask = (abs(stock_sums - 1.0) < 0.01)
                    print(f"Valid ternary points (sum≈1): {valid_mask.sum()}")
                    
                    # Show first few data points
                    print("First 5 data points:")
                    for i in range(min(5, len(falldf_analysis))):
                        a_val, i_val, s_val = falldf_analysis.iloc[i][['stock_a', 'stock_i', 'stock_s']]
                        print(f"  Row {i}: a={a_val:.3f}, i={i_val:.3f}, s={s_val:.3f}, sum={a_val+i_val+s_val:.3f}")

                    # Build raw ternary coordinates to send to the frontend
                    valid_data = falldf_analysis[valid_mask]
                    if len(valid_data) > 0:
                        ternary_points = {
                            'a': valid_data['stock_a'].tolist(),
                            'b': valid_data['stock_i'].tolist(),
                            'c': valid_data['stock_s'].tolist(),
                        }
                    else:
                        # Fallback to all points if no rows pass the mask
                        ternary_points = {
                            'a': stock_a.tolist(),
                            'b': stock_i.tolist(),
                            'c': stock_s.tolist(),
                        }

                    recommended_dist_payload = {
                        'a': float(recommendeddist[0]),  # Angostura
                        'b': float(recommendeddist[1]),  # Isleta
                        'c': float(recommendeddist[2]),  # San Acacia
                    }

                    recommendations['ternary_points'] = ternary_points
                    recommendations['recommended_dist'] = recommended_dist_payload
                    
                    # Plot the histogram of Jul-Aug-Sep catches from simulation and mark the input catch in it for each reach 
                    fig2, axs3 = plt.subplots(1, 3, figsize=(20, 8))
                    axs3 = axs3.flatten()  # Flatten for easier indexing
                    
                    reaches = ['angostura', 'isleta', 'san-acacia']
                    reachnames = ['Angostura', 'Isleta', 'San Acacia']
                    reachidx = ['a','i','s']
                    for i, reach in enumerate(reaches):
                        axs3[i].hist(np.log10(falldf_analysis[f'catch_jul+aug+sep_{reachidx[i]}']+1), bins=30, color='skyblue', edgecolor='black')
                        axs3[i].axvline(x=(np.log10(data[f'dist-total-catch-JAS-{reach}']+1)), color='red', linestyle='--', label='Current Input Catch', linewidth=2)
                        axs3[i].set_title(f'Catch Distribution for {reachnames[i]}', fontsize=16)
                        axs3[i].set_xlabel('Log Catch (Jul + Aug + Sep)', fontsize=14)
                        axs3[i].set_ylabel('Frequency', fontsize=14)
                        axs3[i].legend()
                    
                    plt.tight_layout()

                    # Save histogram plot to base64 string
                    img_buffer3 = io.BytesIO()
                    plt.savefig(img_buffer3, format='png', dpi=150, bbox_inches='tight')
                    img_buffer3.seek(0)
                    histogram_base64 = base64.b64encode(img_buffer3.getvalue()).decode('utf-8')
                    plt.close()
                    
                    # Add plots to recommendations
                    recommendations['histogram_data'] = f"data:image/png;base64,{histogram_base64}"

                except FileNotFoundError:
                    print(f"WARNING: Distribution action CSV not found at {distaction_path}")
                    falldf_analysis = None
                except Exception as e:
                    print(f"ERROR loading distribution CSV or creating plot: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    falldf_analysis = None
                except Exception as e:
                    print(f"ERROR loading distribution CSV: {str(e)}")
                    falldf_analysis = None



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