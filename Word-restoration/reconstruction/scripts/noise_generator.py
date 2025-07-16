import random, cv2
import numpy as np
from PIL import Image, ImageEnhance

def to_pil(cv2_img):
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

def to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def apply_noise(image, noise_type):
    if image is None:
        print(f"Warning: Could not read image. Skipping noise type: {noise_type}")
        return None

    pil_img = to_pil(image)
    cv2_img = image.copy()
    h, w = cv2_img.shape[:2]

    if noise_type == "blur":
        return cv2.GaussianBlur(cv2_img, (3, 3), 0)  

    elif noise_type == "motion blur":
        kernel = np.zeros((9, 9))                   
        kernel[4, :] = np.ones(9)
        kernel /= 9
        return cv2.filter2D(cv2_img, -1, kernel)

    elif noise_type == "pixelation":
        h, w = cv2_img.shape[:2]
        temp = cv2.resize(cv2_img, (w//12, h//12), interpolation=cv2.INTER_LINEAR)  
        return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

    elif noise_type == "compression artifacts":
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]
        _, enc_img = cv2.imencode('.jpg', cv2_img, encode_param)
        return cv2.imdecode(enc_img, 1)

    elif noise_type == "Gaussian noise":
        row, col, ch = cv2_img.shape
        mean = 0
        sigma = 35
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = np.clip(gauss, -sigma*3, sigma*3)
        noisy_img = cv2_img.astype(np.float32) + gauss
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        return noisy_img

    elif noise_type == "low resolution":
        small = cv2.resize(cv2_img, (32, 32), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (cv2_img.shape[1], cv2_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    elif noise_type == "defocus blur":
        return cv2.GaussianBlur(cv2_img, (21, 21), 0)

    elif noise_type == "ringing artifacts":
        blurred = cv2.GaussianBlur(cv2_img, (0, 0), 3)
        return cv2.addWeighted(cv2_img, 1.5, blurred, -0.5, 0)

    elif noise_type == "lighting variations":
        enhancer = ImageEnhance.Brightness(pil_img)
        return to_cv2(enhancer.enhance(random.uniform(0.5, 1.5)))

    elif noise_type == "uneven illumination":
        h, w = cv2_img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        radius = w // 4 
        cv2.circle(mask, (w // 3, h // 3), radius, 255, -1)
        mask_3_channel = cv2.merge([mask]*3)
        return cv2.addWeighted(cv2_img, 1, mask_3_channel, 0.3, 0)

    elif noise_type == "glare":
        glare = np.zeros_like(cv2_img)
        center = (random.randint(0, w), random.randint(0, h))
        radius = random.randint(50, 100)
        cv2.circle(glare, center, radius, (180, 180, 180), -1)  
        return cv2.addWeighted(cv2_img, 0.7, glare, 0.3, 0)

    elif noise_type == "underexposure":
        return cv2.convertScaleAbs(cv2_img, alpha=0.5, beta=-50)

    elif noise_type == "ambient noise":
        ambient = cv2.GaussianBlur(cv2_img, (15, 15), 20)  
        return cv2.addWeighted(cv2_img, 0.6, ambient, 0.4, 0)  


    elif noise_type == "perspective distortion":
        pts1 = np.float32([[0,0], [w,0], [0,h], [w,h]])
        pts2 = np.float32([[random.randint(0, w//10), random.randint(0, h//10)],
                           [w-random.randint(0, w//10), random.randint(0, h//10)],
                           [random.randint(0, w//10), h-random.randint(0, h//10)],
                           [w-random.randint(0, w//10), h-random.randint(0, h//10)]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(cv2_img, M, (w, h))

    elif noise_type == "fading":
        alpha = np.linspace(1, 0.05, w)
        mask = np.tile(alpha, (h, 1))
        return (cv2_img * mask[:, :, np.newaxis]).astype(np.uint8)

    elif noise_type == "ink bleed-through":
        flipped = cv2.flip(cv2_img, 1)
        blended = cv2.addWeighted(cv2_img, 0.8, flipped, 0.2, 0)
        return blended

    elif noise_type == "text smearing":
        kernel = np.ones((1, 6), np.uint8)
        return cv2.dilate(cv2_img, kernel, iterations=2) 

    elif noise_type == "stroke breaks":
        prob = 0.92 
        mask = (np.random.rand(h, w, 1) < prob).astype(np.uint8)
        mask_3_channel = cv2.merge([mask[:, :, 0]] * 3)
        return cv2_img * mask_3_channel

    elif noise_type == "partial occlusion":
        noisy_img = cv2_img.copy()
        for _ in range(2):
            x1, y1 = random.randint(0, max(0, w-50)), random.randint(0, max(0, h-50))
            x2, y2 = x1 + random.randint(20, 35), y1 + random.randint(20, 35)
            cv2.rectangle(noisy_img, (x1, y1), (x2, y2), (0, 0, 0), -1)
        return noisy_img


    elif noise_type == "font erosion":
      kernel = np.ones((3, 3), np.uint8)  
      return cv2.erode(cv2_img, kernel, iterations=2)  


    elif noise_type == "character merging":
        kernels = [np.ones((2, 2), np.uint8), np.ones((3, 3), np.uint8)]
        iterations_options = [1, 2]

        kernel = random.choice(kernels)
        iterations = random.choice(iterations_options)

        return cv2.dilate(cv2_img, kernel, iterations=iterations)

    elif noise_type == "JPEG compression":
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 5]
        _, enc_img = cv2.imencode('.jpg', cv2_img, encode_param)
        decoded_img = cv2.imdecode(enc_img, cv2.IMREAD_COLOR)
        return decoded_img if decoded_img is not None else cv2_img

    elif noise_type == "color space mismatch":
        return cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HSV)

    elif noise_type == "resolution scaling":
        small = cv2.resize(cv2_img, (w//2, h//2), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    elif noise_type == "aliasing":
        small = cv2.resize(cv2_img, (w//3, h//3), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    elif noise_type == "banding":
        banded = cv2_img.copy()
        for i in range(0, h, 20):
            start_row = i
            end_row = min(i + 5, h)
            banded[start_row:end_row] = banded[start_row:end_row] // 3
        return banded

    elif noise_type == "scan line artifacts":
        scan_img = cv2_img.copy()
        for i in range(0, h, 5):
             start_row = i
             end_row = min(i + 1, h)
             scan_img[start_row:end_row] = scan_img[start_row:end_row] // 3
        return scan_img

    elif noise_type == "scanner streaks":
      streak = cv2_img.copy()
      for _ in range(12): 
          x = random.randint(0, w - 3)
          color = [random.randint(0, 255) for _ in range(3)]
          streak[:, x:x+2] = color
      return streak

    elif noise_type == "printer toner gaps":
      toner = cv2_img.copy()
      center = h // 2
      for _ in range(3):
          offset = random.randint(-30, 30)  
          y = np.clip(center + offset, 0, max(0, h - 5))  
          toner[y:y+5, :] = [255, 255, 255]  
      return toner


    elif noise_type == "paper texture interference":
      noise = np.random.normal(127, 50, cv2_img.shape).astype(np.uint8)  # Higher contrast noise
      return cv2.addWeighted(cv2_img, 0.7, noise, 0.3, 0)  # Stronger blend of noise


    elif noise_type == "multi-layer noise":
        noisy = apply_noise(cv2_img, "Gaussian noise")
        if noisy is not None:
          noisy = apply_noise(noisy, "text smearing")
        if noisy is not None:
          return apply_noise(noisy, "JPEG compression")
        return cv2_img
    
    elif noise_type == "watermarks":
      font_scale = max(1.5, h / 50)
      thickness = max(2, w // 150)
      watermark = cv2.putText(cv2_img.copy(), "IIT-J", (w // 4, h // 2),
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (80, 80, 80), thickness, cv2.LINE_AA)
      return cv2.addWeighted(cv2_img, 0.75, watermark, 0.25, 0)  # Heavier watermark blend



    if noise_type == "salt and pepper":
        prob = 0.01  
        noisy = cv2_img.copy()
        black = np.random.rand(h, w) < prob
        white = np.random.rand(h, w) < prob
        noisy[black] = 0
        noisy[white] = 255
        return noisy


    elif noise_type == "ink spread":
        kernel = np.ones((3, 3), np.uint8)  
        return cv2.dilate(cv2_img, kernel, iterations=1) 

    elif noise_type == "horizontal tear":
        y = random.randint(h // 3, 2 * h // 3)
        cv2.line(cv2_img, (0, y), (w, y), (255, 255, 255), thickness=7)
        return cv2_img

    elif noise_type == "vertical tear":
        x = random.randint(w // 3, 2 * w // 3)
        cv2.line(cv2_img, (x, 0), (x, h), (255, 255, 255), thickness=7)
        return cv2_img

    elif noise_type == "smudge":
        smudge_kernel = (9, 9)
        smudged = cv2.blur(cv2_img, smudge_kernel)
        return cv2.addWeighted(cv2_img, 0.6, smudged, 0.4, 0)

    elif noise_type == "text ghosting":
        shifted = np.roll(cv2_img, 5, axis=1)
        return cv2.addWeighted(cv2_img, 0.7, shifted, 0.3, 0)

    elif noise_type == "double scan":
        shifted = np.roll(cv2_img, random.randint(2, 6), axis=0)
        return cv2.addWeighted(cv2_img, 0.5, shifted, 0.5, 0)

    elif noise_type == "wave distortion":
        distorted = np.zeros_like(cv2_img)
        for i in range(h):
            offset = int(10.0 * np.sin(2 * np.pi * i / 60))
            distorted[i] = np.roll(cv2_img[i], offset, axis=1)
        return distorted


    elif noise_type == "paper curl":
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (0, 0), (w, h), 255, thickness=30)
        darken = cv2.merge([mask // 3] * 3)
        return cv2.subtract(cv2_img, darken)

    elif noise_type == "random scratches":
        scratched = cv2_img.copy()
        h, w = scratched.shape[:2]
        for _ in range(random.randint(3, 8)):
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            x2 = x1 + random.randint(-w//2, w//2)
            y2 = y1 + random.randint(-h//2, h//2)
            color = (255, 255, 255) if random.random() < 0.5 else (0, 0, 0)
            thickness = random.randint(1, 2)
            cv2.line(scratched, (x1, y1), (x2, y2), color, thickness)
        return scratched

    elif noise_type == "scribble":
        scribbled = cv2_img.copy()
        h, w = scribbled.shape[:2]
        for _ in range(random.randint(1, 3)):
            points = np.array([
                [random.randint(0, w), random.randint(0, h)]
                for _ in range(random.randint(4, 7))
            ], np.int32).reshape((-1, 1, 2))
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            thickness = random.randint(1, 2)
            cv2.polylines(scribbled, [points], False, color, thickness, cv2.LINE_AA)
        return scribbled

    elif noise_type == "staple holes":
      noisy_img = cv2_img.copy()
      h, w = noisy_img.shape[:2]

      num_holes = random.randint(1, 7)
      radius = random.randint(5, 10)

      for _ in range(num_holes):
          position = random.choice([
              (random.randint(0, w//10), random.randint(0, h//10)),                  # top-left
              (random.randint(0, w//10), random.randint(h*9//10, h - 1)),            # bottom-left
              (random.randint(w*9//10, w - 1), random.randint(0, h//10)),            # top-right
              (random.randint(w*9//10, w - 1), random.randint(h*9//10, h - 1)),      # bottom-right
              (random.randint(0, w//10), random.randint(h//3, h*2//3)),              # middle-left
              (random.randint(w*9//10, w - 1), random.randint(h//3, h*2//3))         # middle-right
          ])
          cv2.circle(noisy_img, position, radius, (0, 0, 0), -1)

      return noisy_img
    elif noise_type == "water damage":
      h, w = image.shape[:2]
      blurred = cv2.blur(image, (6, 6))
      mask = np.zeros((h, w), dtype=np.uint8)

      num_spots = 2  # Number of damage spots
      for _ in range(num_spots):
          center = (random.randint(0, w), random.randint(0, h))
          radius = random.randint(30, 60)
          cv2.circle(mask, center, radius, 255, -1)

      mask_3ch = cv2.merge([mask]*3)
      return np.where(mask_3ch == 255, blurred, image)

    elif noise_type == "angled line":
      h, w = image.shape[:2]
      noisy_img = image.copy()

      color = (255, 255, 255) if random.random() > 0.5 else (0, 0, 0)
      thickness = 10

      angle = random.uniform(0, 180)
      radians = np.deg2rad(angle)

      # Compute line endpoints so it crosses the image fully
      x0 = int(w / 2 - w * np.cos(radians))
      y0 = int(h / 2 - w * np.sin(radians))
      x1 = int(w / 2 + w * np.cos(radians))
      y1 = int(h / 2 + w * np.sin(radians))

      cv2.line(noisy_img, (x0, y0), (x1, y1), color, thickness)

      return noisy_img
    elif noise_type == "random_erasure":
      h, w = image.shape[:2]
      output = image.copy()
      max_rect = 4;
      num_boxes = random.randint(2,4)  # Always at least 3 boxes

      for _ in range(num_boxes):
          x1 = random.randint(0, max(0, w - 10))
          y1 = random.randint(0, max(0, h - 10))

          box_width = random.randint(7, max(8, w // 5))
          box_height = random.randint(5, max(6, h // 4))

          x2 = min(w, x1 + box_width)
          y2 = min(h, y1 + box_height)

          # Randomly choose color: black or white
          color = 0 if random.random() < 0.5 else 255
          output[y1:y2, x1:x2] = color

      return output

    else:
        return cv2_img
