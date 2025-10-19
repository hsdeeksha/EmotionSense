import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from collections import deque
from collections import Counter
import json
import datetime

# command line argument
ap = argparse.ArgumentParser()
default_model = os.path.join(os.path.dirname(__file__), 'model.h5')
ap.add_argument("--mode", help="train/display")
ap.add_argument("--model", help="path to model .h5 (full model or weights)", default=default_model)
ap.add_argument("--camera", help="camera index (0,1,..)", type=int, default=0)
ap.add_argument("--virtual", help="enable virtual webcam output (pyvirtualcam)", action='store_true')
ap.add_argument("--vcam-fps", help="virtual camera FPS (when --virtual is set)", type=int, default=20)
ap.add_argument("--save-on", help="comma-separated list of emotions to save on (e.g. Happy,Sad). Empty disables.", default="")
ap.add_argument("--save-dir", help="directory to save images/clips", default=os.path.join(os.path.dirname(__file__), '..', 'saved'))
ap.add_argument("--clip-length", help="length of saved clip in seconds (includes pre-clip)", type=float, default=3.0)
ap.add_argument("--pre-clip", help="seconds of video to include before trigger (must be <= clip-length)", type=float, default=1.0)
ap.add_argument("--cooldown", help="minimum seconds between saves for same emotion", type=float, default=5.0)
ap.add_argument("--min-count", help="minimum recent count of the emotion required to trigger save", type=int, default=2)
ap.add_argument("--save-images-only", help="only save still images (no clips)", action='store_true')
args = ap.parse_args()
mode = args.mode
model_arg = args.model
camera_index = args.camera

# plots accuracy and loss curves
def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# If you want to train the same model or try other models, go for this
if mode == "train":
    # training parameters and data generators are created only when training
    train_dir = 'data/train'
    val_dir = 'data/test'

    num_train = 28709
    num_val = 7178
    batch_size = 64
    num_epoch = 50

    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical')

    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
    model_info = model.fit_generator(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size)
    plot_model_history(model_info)
    model.save_weights('model.h5')

# emotions will be displayed on your face from the webcam feed
elif mode == "display":
    # Try loading a full saved model first (saved with model.save()).
    # If that fails, fall back to loading weights into the defined architecture (model.save_weights()).
    model_path = model_arg
    try:
        loaded = load_model(model_path)
        model = loaded
        print('Loaded full model from', model_path)
    except Exception:
        # fall back to weights only
        model.load_weights(model_path)
        print('Loaded weights into architecture from', model_path)

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # keep a short history of recent emotions to display as a bar
    history_len = 60
    emotion_history = deque(maxlen=history_len)
    overlay_enabled = True

    # parse save-on list
    save_on_raw = args.save_on.strip()
    save_on = set([s.strip() for s in save_on_raw.split(',') if s.strip()]) if save_on_raw else set()
    save_dir = os.path.abspath(args.save_dir)
    clip_length = max(0.5, float(args.clip_length))
    pre_clip = min(clip_length, max(0.0, float(args.pre_clip)))
    cooldown = max(0.0, float(args.cooldown))
    min_count = max(1, int(args.min_count))
    save_images_only = bool(args.save_images_only)

    # prepare save directory
    if save_on:
        os.makedirs(save_dir, exist_ok=True)
        # prebuffer frames to include pre-clip when saving short clips
        prebuffer_frames = deque(maxlen=int(pre_clip * args.vcam_fps if args.vcam_fps > 0 else 20))
        # track last saved timestamps per emotion to enforce cooldown
        last_saved = {emo: 0.0 for emo in save_on}

    # mapping to colors for emoji/background (BGR)
    emotion_colors = {
        'Angry': (0, 0, 255),        # red
        'Disgusted': (0, 128, 0),    # dark green
        'Fearful': (0, 255, 255),    # yellow
        'Happy': (0, 215, 255),      # orange-ish
        'Neutral': (200, 200, 200),  # gray
        'Sad': (255, 0, 0),          # blue
        'Surprised': (255, 192, 203) # pink
    }

    def draw_emoji_on_frame(frame, center_x, top_y, size, emotion):
        """Draw a simple emoji above the face using OpenCV primitives.
        frame: BGR image, center_x: int x-coordinate, top_y: y coordinate where emoji bottom should start
        size: diameter in pixels, emotion: string label
        """
        # compute emoji bounding box -- make emoji small relative to face
        diameter = max(24, min(120, int(size) // 2))
        radius = diameter // 2
        cx = int(center_x)
        # put emoji above the face rectangle (top_y is the top of the face rect)
        cy = int(top_y - radius - 8)
        color = emotion_colors.get(emotion, (200,200,200))

        # draw face circle
        cv2.circle(frame, (cx, cy), radius, color, -1)
        # draw eyes
        eye_y = cy - radius//4
        eye_x_offset = max(1, radius//2 - 6)
        eye_radius = max(1, radius//8)
        cv2.circle(frame, (cx - eye_x_offset, eye_y), eye_radius, (0,0,0), -1)
        cv2.circle(frame, (cx + eye_x_offset, eye_y), eye_radius, (0,0,0), -1)
        # draw mouth depending on emotion
        mouth_y = cy + max(2, radius//6)
        if emotion == 'Happy':
            # smile
            cv2.ellipse(frame, (cx, mouth_y), (max(1, radius//2), max(1, radius//3)), 0, 0, 180, (0,0,0), 2)
        elif emotion == 'Sad' or emotion == 'Angry':
            # frown
            cv2.ellipse(frame, (cx, mouth_y+6), (max(1, radius//2), max(1, radius//4)), 0, 180, 360, (0,0,0), 2)
        elif emotion == 'Surprised':
            # open mouth
            cv2.circle(frame, (cx, mouth_y), max(1, radius//6), (0,0,0), -1)
        else:
            # neutral small line
            cv2.line(frame, (cx - radius//4, mouth_y), (cx + radius//4, mouth_y), (0,0,0), 2)

    def draw_history_bar(frame, history, emotions, width=200):
        """Draw a small vertical bar chart showing recent emotions on the right side of the frame."""
        h, w = frame.shape[:2]
        bar_x0 = w - width - 10
        bar_y0 = 10
        bar_h = h - 20
        # draw a simple solid dark panel (no colored stripes) per request
        cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + width, bar_y0 + bar_h), (20,20,20), -1)
        # draw a thin border
        cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + width, bar_y0 + bar_h), (80,80,80), 2)

    # start the webcam feed
    cap = cv2.VideoCapture(camera_index)
    vcam = None
    # output frame size (used for display and optional virtual camera). Use a sensible default
    out_w, out_h = 1280, 720

    # If user requests a virtual camera, try to initialize pyvirtualcam safely.
    if args.virtual:
        try:
            import pyvirtualcam
            from pyvirtualcam import PixelFormat
            # create a virtual camera using RGB pixel format
            vcam = pyvirtualcam.Camera(width=out_w, height=out_h, fps=args.vcam_fps, fmt=PixelFormat.RGB)
            print(f'Virtual camera started: {out_w}x{out_h} @ {args.vcam_fps}fps')
        except Exception as e:
            print('pyvirtualcam not available or failed to start virtual camera:', e)
            print('To enable virtual camera, install pyvirtualcam: pip install pyvirtualcam')
            vcam = None

    # create a resizable window so the user can focus and send key events reliably
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    try:
        while True:
            # Find haar cascade to draw bounding box around face
            ret, frame = cap.read()
            if not ret:
                break
            # load cascade from script directory so it works regardless of current working directory
            cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
            if not os.path.exists(cascade_path):
                raise FileNotFoundError(f'Haar cascade not found at {cascade_path}. Make sure the file exists in the src folder.')
            facecasc = cv2.CascadeClassifier(cascade_path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                # suppress progress logging from predict
                prediction = model.predict(cropped_img, verbose=0)
                maxindex = int(np.argmax(prediction))
                emotion_label = emotion_dict[maxindex]
                # update history
                emotion_history.append(emotion_label)
                # append frame to prebuffer if saving is enabled
                if save_on:
                    # store a shallow copy of the current frame for later writing
                    prebuffer_frames.append(frame.copy())
                # check saving triggers
                if save_on and emotion_label in save_on:
                    # require that the emotion appears at least `min_count` times in recent history
                    if emotion_history.count(emotion_label) >= min_count:
                        now_ts = datetime.datetime.utcnow().timestamp()
                        if now_ts - last_saved.get(emotion_label, 0.0) >= cooldown:
                            # create filenames
                            stamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
                            safe_label = emotion_label.replace(' ', '_')
                            img_name = f"{stamp}_{safe_label}.jpg"
                            img_path = os.path.join(save_dir, img_name)
                            try:
                                # save still image
                                cv2.imwrite(img_path, frame)
                                print('Saved image for', emotion_label, '->', img_path)
                            except Exception as e:
                                print('Failed to save image:', e)
                            # save short clip unless images-only
                            if (not save_images_only) and clip_length > 0.5:
                                try:
                                    # determine frames to write: prebuffer + upcoming frames (write synchronously)
                                    frames_needed = int(round(clip_length * args.vcam_fps))
                                    # collect frames: take prebuffer then the current frame and capture subsequent frames
                                    clip_frames = list(prebuffer_frames)
                                    clip_frames.append(frame.copy())
                                    # capture additional frames from camera (blocking for up to clip length)
                                    cap_start = datetime.datetime.utcnow().timestamp()
                                    while len(clip_frames) < frames_needed:
                                        ret2, f2 = cap.read()
                                        if not ret2:
                                            break
                                        clip_frames.append(f2.copy())
                                        # small sleep to pace capture roughly according to vcam_fps
                                    # write clip via VideoWriter
                                    clip_name = f"{stamp}_{safe_label}.avi"
                                    clip_path = os.path.join(save_dir, clip_name)
                                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                                    fps = args.vcam_fps if args.vcam_fps and args.vcam_fps > 0 else 20
                                    h_f, w_f = clip_frames[0].shape[:2]
                                    writer = cv2.VideoWriter(clip_path, fourcc, fps, (w_f, h_f))
                                    for cf in clip_frames:
                                        writer.write(cf)
                                    writer.release()
                                    print('Saved clip for', emotion_label, '->', clip_path)
                                except Exception as e:
                                    print('Failed to save clip:', e)
                            last_saved[emotion_label] = now_ts
                if overlay_enabled:
                    # draw label only (emoji removed)
                    cv2.putText(frame, emotion_label, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # draw history bar and stats if overlay enabled
            if overlay_enabled:
                # draw live stats top-left
                counts = Counter(emotion_history)
                total = max(1, len(emotion_history))
                # compact stats box (narrower, smaller line spacing)
                box_x = 8
                box_y = 8
                box_w = 180
                # one title line + up to 5 entries
                max_lines = 5
                line_h = 18
                n = min(max_lines, len(counts))
                box_h = line_h * (1 + n) + 12
                box_x1 = box_x + box_w
                # background box
                cv2.rectangle(frame, (box_x, box_y), (box_x1, box_y + box_h), (0,0,0), -1)
                # title
                title_y = box_y + line_h
                cv2.putText(frame, 'Live stats (recent)', (box_x + 6, title_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
                # entries: draw label left, percentage right-aligned
                i = 0
                for k, v in counts.most_common(max_lines):
                    pct = int((v/total)*100)
                    label = k
                    pct_text = f"{pct}%"
                    y = box_y + line_h * (i + 2) - 2
                    cv2.putText(frame, label, (box_x + 6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
                    (pw, ph), _ = cv2.getTextSize(pct_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    px = box_x1 - 6 - pw
                    cv2.putText(frame, pct_text, (px, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
                    i += 1

            out_frame = cv2.resize(frame,(out_w,out_h),interpolation = cv2.INTER_CUBIC)
            cv2.imshow('Video', out_frame)
            if vcam is not None:
                # convert BGR to RGB and send to virtual cam
                try:
                    send = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
                    vcam.send(send)
                    vcam.sleep_until_next_frame()
                except Exception:
                    # don't crash the main loop if virtual camera send fails
                    pass
            key = cv2.waitKey(1)
            # Accept 'q' or 'Q' or ESC to quit
            if key != -1:
                k = key & 0xFF
                # toggle overlays with 'o' or 'O'
                if k == ord('o') or k == ord('O'):
                    overlay_enabled = not overlay_enabled
                    continue
                if k == ord('q') or k == ord('Q') or k == 27:
                    break
    except KeyboardInterrupt:
        # allow Ctrl+C from terminal to stop the loop
        pass
    finally:
        if vcam is not None:
            try:
                vcam.close()
            except Exception:
                pass
        cap.release()
        cv2.destroyAllWindows()
        # save summary of recent emotions
        try:
            if len(emotion_history) > 0:
                counts = Counter(emotion_history)
                total = len(emotion_history)
                summary = {
                    'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
                    'total_samples': total,
                    'counts': counts
                }
                out_path = os.path.join(os.path.dirname(__file__), '..', 'emotion_summary.json')
                with open(out_path, 'w', encoding='utf-8') as f:
                    # convert Counter to dict for JSON
                    summary['counts'] = dict(counts)
                    json.dump(summary, f, indent=2)
                print('Saved emotion summary to', out_path)
        except Exception:
            pass