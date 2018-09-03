import cv2 
import face_recognition

capture_video_frame = cv2.VideoCapture(0)

input_image_forref = face_recognition.load_image_file("hk.jpg")
input_face_encoding = face_recognition.face_encodings(input_image_forref)[0]

known_faces = [
    input_face_encoding
]

face_names_according_to_index = [
    "Harikrishna"
]

face_polygon = []
face_encoding = []
face_names = []
frame_process_buf = True

while True:

    readf, frame = capture_video_frame.read()

    crop_frame = cv2.resize(frame, (0,0), fx=1/4, fy=1/4)

    frame_in_rgb = crop_frame[:,:, ::-1]

    if frame_process_buf:
        
        face_polygon = face_recognition.face_locations(frame_in_rgb)
        face_encodings = face_recognition.face_encodings(frame_in_rgb, face_polygon)

        face_names = []

        for face_polygons in face_encodings:

            match = face_recognition.compare_faces(known_faces, face_polygons)
            
            unid = "UnIdentified Face"

            if True in match:
                match_first_index = match.index(True)
                unid = face_names_according_to_index[match_first_index]
            face_names.append(unid)
    
    frame_process_buf = not frame_process_buf


    for(ftop, fright, fbottom, fleft), unid in zip(face_polygon, face_names):

        ftop *= 4
        fright *=4
        fbottom *= 4
        fleft *= 4

        cv2.rectangle(frame,(fleft,ftop), (fright,fbottom), (255,0,0), 1)
        cv2.rectangle(frame, (fleft, fbottom), (fright, fbottom), (255, 0, 0), cv2.FILLED)
        font = cv2.FONT_ITALIC
        cv2.putText(frame,unid, (fleft+10, fbottom+10), font, 0.5, (255,255,255),1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture_video_frame.release()
cv2.destroyAllWindows()        
