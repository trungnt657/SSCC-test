import glob
import os
import commpy
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToPILImage
from commpy.channelcoding.ldpc import get_ldpc_code_params, ldpc_bp_decode, triang_ldpc_systematic_encode

# --- User-configurable paths ---
# Directory containing input images (e.g., 'image_test')
INPUT_DIR = 'images'
# Directory to save JPEG-compressed images at target BPP
JPEG_OUTPUT_DIR = os.path.join('2JPEG_rayleigh', 'JPEG_images')
# Directory to save bittext files for JPEG images
BITTEXT_DIR = os.path.join('2JPEG_rayleigh', 'JPEG_bit')
# Directory to save channel-processed JPEG images
CHANNEL_OUTPUT_DIR = os.path.join('2JPEG_rayleigh', 'JPEG_7dB')
# Path to LDPC design file (parity-check matrix)
LDPC_DESIGN_FILE = '/home/ser/SSCC-test/SSCC/Image/BPG&JPEG&WebP+LDPC+QAM+AWGN&Ray&Rician-Python/1440.720.txt'


def pillow_encode(img, out_path, fmt='JPEG', quality=10):
    img.save(out_path, format=fmt, quality=quality)
    size = os.path.getsize(out_path)
    return size * 8.0 / (img.width * img.height)


def find_closest_bpp(target, img, temp_path, fmt='JPEG'):
    low, high = 0, 100
    prev = None
    bpp = None
    for _ in range(10):
        mid = (low + high) // 2
        if prev is not None and mid == prev:
            break
        prev = mid
        bpp = pillow_encode(img, temp_path, fmt, quality=mid)
        if bpp > target:
            high = mid - 1
        else:
            low = mid + 1
    return bpp


def img_to_bittext(img_path, txt_path):
    with open(img_path, 'rb') as f:
        data = f.read()
    bits = ''.join(f"{b:08b}" for b in data)
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, 'w') as f:
        f.write(bits)


def get_bitarray(txt_path):
    with open(txt_path, 'r') as f:
        return np.array([int(b) for b in f.read().strip()], dtype=np.uint8)


def random_noise(nc, w, h):
    return ToPILImage()(torch.rand(nc, w, h))


def bittext_to_img(bits, orig_jpeg_path, out_dir):
    # Accept bits as string or array/list
    if not isinstance(bits, str):
        bits = ''.join(str(int(b)) for b in np.array(bits).flatten())

    # Reconstruct raw bytes
    byte_vals = [int(bits[i:i+8], 2) for i in range(0, len(bits), 8)]
    data = bytearray(byte_vals)

    sub = os.path.basename(os.path.dirname(orig_jpeg_path))
    target_dir = os.path.join(out_dir, sub)
    os.makedirs(target_dir, exist_ok=True)
    out_path = os.path.join(target_dir, os.path.basename(orig_jpeg_path))

    with open(out_path, 'wb') as f:
        f.write(data)

    try:
        Image.open(out_path).convert('RGB')
    except IOError:
        w, h = Image.open(orig_jpeg_path).size
        random_noise(3, w, h).save(out_path)


def ldpc_qam_awgn(input_bits, snr=2, qam_order=16):
    # LDPC encode
    param = get_ldpc_code_params(LDPC_DESIGN_FILE)
    coded = triang_ldpc_systematic_encode(input_bits, param)

    # QAM modulate
    modem = commpy.QAMModem(qam_order)
    sym = modem.modulate(coded.reshape(-1))

    # Rayleigh + AWGN
    N = len(sym)
    h = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
    y = commpy.awgn(sym * h, snr)
    y_eq = y / h

    # QAM demodulate
    dem = modem.demodulate(y_eq, 'hard').reshape(coded.shape)

    # Prepare LLR: map 1 -> -1, 0 -> +1
    llr = dem.astype(float)
    llr[llr == 1] = -1
    llr[llr == 0] = 1
    llr_seq = llr.reshape(-1)

    # LDPC decode
    decoded_seq, _ = ldpc_bp_decode(llr_seq, param, 'MSA', 10)
    # Truncate to original message length
    decoded = decoded_seq[:len(input_bits)]
    return np.array(decoded, dtype=np.uint8)

def bit_to_img(input_bits, orig_jpeg_path, out_dir, snr=2, qam_order=16):
    # LDPC + QAM + AWGN
    rec_bits = ldpc_qam_awgn(input_bits, snr=snr, qam_order=qam_order)
    bittext_to_img(rec_bits, orig_jpeg_path, out_dir)
    return rec_bits

if __name__ == '__main__':
    fmt = 'jpg'
    target_bpp = 0.9
    snr = 7
    qam_order = 4

    # 1) JPEG compress to target BPP
    pattern = os.path.join(INPUT_DIR, '**', f'*.{fmt}')
    imgs = glob.glob(pattern, recursive=True)
    if not imgs:
        raise RuntimeError(f"No .{fmt} images found in {INPUT_DIR}")

    total_bpp = 0
    for img_path in imgs:
        sub = os.path.basename(os.path.dirname(img_path))
        out_dir = os.path.join(JPEG_OUTPUT_DIR, sub)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, os.path.splitext(os.path.basename(img_path))[0] + '.JPEG')
        img = Image.open(img_path).convert('RGB')
        bpp = find_closest_bpp(target_bpp, img, out_file)
        total_bpp += bpp
        print(f"{img_path}: {bpp:.4f} bpp")
    print(f"平均bpp: {total_bpp/len(imgs):.4f}")

    # 2) Convert JPEGs to bit-text
    jpegs = glob.glob(os.path.join(JPEG_OUTPUT_DIR, '**', '*.JPEG'), recursive=True)
    for jpg in jpegs:
        sub = os.path.basename(os.path.dirname(jpg))
        txt_dir = os.path.join(BITTEXT_DIR, sub)
        os.makedirs(txt_dir, exist_ok=True)
        txt_file = os.path.join(txt_dir, os.path.splitext(os.path.basename(jpg))[0] + '.txt')
        img_to_bittext(jpg, txt_file)
        print(f"{jpg} -> {txt_file}")

    # 3) Channel coding + rebuild
    txts = glob.glob(os.path.join(BITTEXT_DIR, '**', '*.txt'), recursive=True)
    for txt in txts:
        bits = get_bitarray(txt)
        rec = ldpc_qam_awgn(bits, snr=snr, qam_order=qam_order)
        bittext_to_img(rec, os.path.join(JPEG_OUTPUT_DIR,
                                         os.path.basename(os.path.dirname(txt)),
                                         os.path.splitext(os.path.basename(txt))[0] + '.JPEG'),
                       CHANNEL_OUTPUT_DIR)
        print(f"Processed {txt}")
