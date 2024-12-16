

```markdown
# Real-Time Audio Processing and Classification

This project implements a real-time audio classification system that can detect specific sounds such as gunshots, screams, and shattering glass. It includes features like live audio recording, feature extraction, emotion detection, and alert notifications via email.

## Features
- Real-time audio recording and processing.
- Detection of critical sounds (e.g., gunshots, screams).
- Automated email alerts for critical detections.
- Pre-trained model integration.

## Prerequisites
### Python Libraries
Ensure you have the following Python libraries installed:
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Librosa
- SoundDevice
- Seaborn
- tqdm
- scikit-learn
- keyboard
- argparse

Install dependencies using pip:
```bash
pip install -r requirements.txt
```

### Data Files
- Pre-trained model files (`model_path`).
- Normalization files (`mean_new1002.npy` and `std_new1002.npy`).

### Optional
To enable email alerts, configure your SMTP server credentials in the `send_email_alert` function.

## Usage
### 1. Running Real-Time Audio Processing
Execute the script to start real-time audio processing:
```bash
python audio_sentiment_prediction.py --mode real_time

```

### 2. Predicting from Audio Files
Provide the path to an audio file for prediction:
```bash
python audio_sentiment_prediction.py --mode recorded --audio_file C:\Users\Vaishnave\Documents\Audio_Sentiment_Analysis\Audio-Sentiment-Analysis\aud_data\02\03-01-02-01-01-01-02.wav

```


### 3. Detect Critical Sounds
The script detects and classifies the following sounds:
- `Gunshot_or_gunfire`
- `Screams`
- `Shatter`
- `background_noise`

Critical sounds trigger an email alert to the recipients specified in the script.

### 5. Interrupt the Process
To stop real-time recording, press `Ctrl+C`.

## File Structure
- `real_time_vandalism_detection.py`: Main script to execute.
- `mean_new1002.npy`: File containing mean values for normalization.
- `std_new1002.npy`: File containing standard deviation values for normalization.
- `model_path`: Path to the pre-trained TensorFlow model.


## Known Issues
- Ensure the audio device is connected and configured properly for recording.
- Configure email credentials for alerts if needed.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
```

Update the placeholders like `your_script.py`, `model_path`, and email credentials as needed.
