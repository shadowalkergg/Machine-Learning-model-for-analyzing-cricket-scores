import json
import glob
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

file_path = r"C:\Users\gupta\Downloads\t20s_male_json"
files = os.listdir(file_path)

def process_cricket_json(file_path):
    with open(file_path) as f:
        data =json.load(f)

    match_id=file_path.split('/')[-1]
    venue=data['info']['venue']

    deliveries=data['innings'][0]['overs']

    current_score=0
    wickets=0

    stats_at_15=None

    for over_data in deliveries:
        over_num=over_data['over']
        for delivery in over_data['deliveries']:
            runs=delivery['runs']['total']
            current_score+=runs
            if 'wickets' in delivery:
                wickets+=len(delivery['wickets'])
        if over_num==14:
            stats_at_15={'score':current_score,'wickets':wickets}
    final_score=current_score

    if stats_at_15:
        return {
            'venue':venue,
            'score_at_15': stats_at_15['score'],
            'wickets_at_15': stats_at_15['wickets'],
            'final_score': final_score
        }
    return None

all_data = []
for file in glob.glob(os.path.join(file_path, "*.json")):
    result = process_cricket_json(file)
    if result:
        all_data.append(result)

df = pd.DataFrame(all_data)
#print(df.head())

le=LabelEncoder()
df['venue_encoded']=le.fit_transform(df['venue'])

x=df[['venue_encoded', 'score_at_15', 'wickets_at_15']]
y=df['final_score']

scaler=StandardScaler()
X_scaled=scaler.fit_transform(x)
X_train,X_test,Y_train,Y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)

model=SVR(kernel='rbf',C=100,gamma=0.2)
model.fit(X_train,Y_train)

predictions=model.predict(X_test)
error=mean_absolute_error(Y_test,predictions)

print(f"Average Error: {error:2f} runs")
import matplotlib.pyplot as plt


plt.figure(figsize=(10, 6))
plt.scatter(Y_test, predictions, alpha=0.7, color='blue')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2) # Diagonal line
plt.xlabel('Final Score')
plt.ylabel('Predicted score')
plt.title('Comparison between predicted and the actual score')
plt.show()
def predict_final_score(venue, score_15, wickets_15):
    try:
        venue_encoded = le.transform([venue])
    except:
        venue_encoded = [0]
    input_data = pd.DataFrame({
        'venue_encoded': venue_encoded,
        'score_at_15': [score_15],
        'wickets_at_15': [wickets_15]
    })
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    return int(prediction[0])
print("\nLive Cricket Score Predictor")
while True:
    v = input("Enter Venue (or 'exit'): ")
    if v == 'exit': break
    s = int(input("Score at 15 overs: "))
    w = int(input("Wickets lost: "))

    result = predict_final_score(v, s, w)
    print(result)   