import gradio as gr
import random

# def make_prediction(test_input, model):
#     model.predict(test_input)

def predict(*args):
    attackDir,Home_no, Home_y=0,0,1
    openPlay, setPlay = 1,1

    scoreH, scoreA = args[0],args[1]
    timeL = float(90*60 - args[2]*60 - args[3])
    if args[4]=="Left":
        attackDir=-1
    L_x = args[5]
    L_y = args[6]
    if args[7]=='Yes':
        Home_y=1
    if args[8]=='Set Play':
        openPlay=0
    elif args[8]=='Open Play':
        setPlay=0
    
    import pandas as pd
    test = pd.DataFrame(data=[scoreH, scoreA, timeL, attackDir,L_x, L_y,Home_no, Home_y, openPlay, setPlay]).T
    
    from tensorflow import keras
    model = keras.models.load_model('model_96.8.h5')
    prob = model.predict(test,verbose=False)
    return str(round((prob[0][0]),5))+" probability of scoring a goal"

unique_hometeam = ['Yes','No']  
unique_phasetype = ['Set Play', 'Open Play']
unique_attackingdirection = ['Right', 'Left']


with gr.Blocks(css=".gradio-container {background-image: url('file=wallpaper.webp')}") as demo:
    gr.Markdown("""
    **Goal Classification**:  This demo uses a DNN classifier to predict probability based on environmental and mentality-affecting factor.
    """)

    with gr.Row():
        with gr.Column():
            Mintime = gr.Slider(label='Minutes into the game', minimum=0, maximum=90,step=1)
            Sectime = gr.Slider(label='Seconds', minimum=0, maximum=60,step=1)
            GoalHome = gr.Slider(label='Goals scored by your team', minimum=0, maximum=20,step=1)
            GoalAway = gr.Slider(label='Goals scored by opposing team', minimum=0, maximum=20,step=1)

            HomeTeam = gr.Dropdown(
                label="Are you playing on home ground?",
                choices=unique_hometeam,
                # value=lambda: random.choice(unique_hometeam),
            )

            PhaseType = gr.Dropdown(
                label="What kind of play is going on?",
                choices=unique_phasetype,
                # value=lambda: random.choice(unique_phasetype),
            )
            AttackingDirection = gr.Dropdown(
                label="Scoring direction",
                choices=unique_attackingdirection,
                # value=lambda: random.choice(unique_attackingdirection),
            )

            x_coordinate = gr.Slider(label='X-coordinate (-ve if Attacking Direction is Left)', minimum=-5420, maximum=5420,default=0)
            y_coordinate = gr.Slider(label='Y-coordinate', minimum=-4640, maximum=4640,step=1,default=0)

        with gr.Column():
            label = gr.Label()
            with gr.Row():
                predict_btn = gr.Button(value="Predict")
            predict_btn.click(predict,inputs=[GoalHome,GoalAway,Mintime,Sectime,AttackingDirection,x_coordinate,y_coordinate,HomeTeam,PhaseType],outputs=label)

demo.launch(share=True)