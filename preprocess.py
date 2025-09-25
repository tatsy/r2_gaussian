import re
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import tigre
import tigre.algorithms as algs
from omegaconf import OmegaConf


def main(args: argparse.Namespace):
    # JSON configuration
    json_file = Path(args.input) / 'config.json'
    if not json_file.exists():
        raise FileNotFoundError(f'{str(json_file):s} not found!')
    meta = OmegaConf.create(json.loads(json_file.read_text()))
    print('Matadata:')
    print(json.dumps(dict(meta), indent=4))
    print('')

    # Read the input file
    target_dir = Path(args.input)
    output_dir = target_dir / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(r'.*(\d+)\.tif')
    files = target_dir.rglob('*.tif')
    files = sorted([f for f in files if pattern.match(f.name)])
    print(f'Found {len(files)} files')

    if args.n_views > 0:
        skip = len(files) // args.n_views
    else:
        skip = 1

    train_set = set(range(0, len(files), skip))
    train_proj = sorted(list(train_set))
    test_proj = sorted([i for i in range(len(files)) if i not in train_set])
    angles = np.linspace(0, 2.0 * np.pi, len(files), endpoint=False)

    images = []
    for i in range(len(files)):
        image = cv2.imread(str(files[i]), cv2.IMREAD_UNCHANGED)
        assert image is not None, f'Failed to read {str(files[i]):s}'
        assert image.dtype == np.uint16, f'Unsupported image type: {image.dtype}'

        org_height, org_width = image.shape
        if args.resize > 0:
            image = cv2.resize(image, (args.resize, args.resize), interpolation=cv2.INTER_LINEAR)

        images.append(image)

        height, width = image.shape
        pixel_scale = 0.5 * (org_height / height + org_width / width)
        pixel_size = meta.pixel_size * pixel_scale

    images = np.stack(images, axis=0)
    if meta.clockwise:
        images = images[::-1]

    train_angles = angles[train_proj]
    test_angles = angles[test_proj]

    sino = images.astype(np.float32, copy=False)
    sino = sino + 1.0
    scale = 1.0 / sino.max() * 1.005
    sino = -np.log(sino * scale)
    sino = np.ascontiguousarray(sino)

    for i in train_proj:
        img_file = output_dir / f'proj_{i:04d}.npy'
        np.save(str(img_file), sino[i])

    for i in test_proj:
        img_file = output_dir / f'proj_{i:04d}.npy'
        np.save(str(img_file), sino[i])

    geo = tigre.geometry(mode='cone', default=True)
    geo.filter = 'shepp_logan'
    geo.DSD = meta.sdd
    geo.DSO = meta.sod
    geo.nDetector = np.array([sino.shape[1], sino.shape[2]])
    geo.dDetector = np.array([pixel_size, pixel_size])
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.nVoxel = np.array([args.n_voxels, args.n_voxels, args.n_voxels])

    delta = pixel_size * meta.sod / meta.sdd
    geo.sVoxel = np.array([delta * args.n_voxels, delta * args.n_voxels, delta * args.n_voxels])
    geo.dVoxel = geo.sVoxel / geo.nVoxel

    qualmeas = ['RMSE', 'SSD']
    sino_train = np.ascontiguousarray(sino[train_proj])
    if args.tigre_algo.lower() == 'fdk':
        ct = algs.fdk(sino_train, geo, train_angles)
    elif args.tigre_algo.lower() == 'sart':
        ct, _ = algs.ossart(
            sino_train,
            geo,
            train_angles,
            args.tigre_iters,
            lmbda=1.0,
            lmbda_red=0.999,
            verbose=True,
            Quameasopts=qualmeas,
            computel2=True,
            blocksize=10,
            OrderStrategy='random',
        )
    elif args.tigre_algo.lower() == 'sart_tv':
        ct, _ = algs.ossart_tv(
            sino_train,
            geo,
            train_angles,
            args.tigre_iters,
            lmbda=1.0,
            lmbda_red=0.999,
            alpha=0.02,
            alpha_red=0.95,
            verbose=True,
            Quameasopts=qualmeas,
            computel2=True,
            blocksize=10,
            OrderStrategy='random',
        )
    elif args.tigre_algo.lower() == 'os_asd_pocs':
        ct, _ = algs.os_asd_pocs(
            sino_train,
            geo,
            train_angles,
            args.tigre_iters,
            lmbda=1.0,
            lmbda_red=0.999,
            alpha=0.02,
            alpha_red=0.95,
            verbose=True,
            Quameasopts=qualmeas,
            computel2=True,
            blocksize=10,
            OrderStrategy='random',
        )

    ct_file = output_dir / 'vol.npy'
    np.save(str(ct_file), ct)
    print(f'CT image saved to {str(ct_file):s}')

    conf = OmegaConf.create({})
    conf.vol = ct_file.name
    conf.scanner = {}
    conf.proj_train = [{'angle': float(a), 'file_path': f'proj_{i:04d}.npy'} for a, i in zip(train_angles, train_proj)]
    conf.proj_test = [{'angle': float(a), 'file_path': f'proj_{i:04d}.npy'} for a, i in zip(test_angles, test_proj)]

    geo_attribs = [
        'nVoxel',
        'sVoxel',
        'dVoxel',
        'nDetector',
        'sDetector',
        'dDetector',
        'DSO',
        'DSD',
        'offDetector',
        'offOrigin',
        'mode',
        'filter',
        'accuracy',
    ]
    for key in geo_attribs:
        val = getattr(geo, key)
        if isinstance(val, (int, float, str, list, tuple)):
            conf.scanner[key] = val
        elif isinstance(val, np.ndarray):
            conf.scanner[key] = val.tolist()
        else:
            raise TypeError(f'Unsupported type: {type(val)} for geo.{key}')

    with open(output_dir / 'meta_data.json', 'w') as f:
        json.dump(OmegaConf.to_container(conf, resolve=True), f, indent=2)

    print(f'Metadata saved to {str(output_dir / "meta_data.json"):s}')

    ref_img = ct[ct.shape[0] // 2]
    ref_img = (ref_img - ref_img.min()) / (ref_img.max() - ref_img.min()) * 255.0
    ref_img = ref_img.astype(np.uint8)
    cv2.imwrite(str(output_dir / 'ref.png'), ref_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, default='output')
    parser.add_argument('--n_voxels', type=int, default=512)
    parser.add_argument('--n_views', type=int, default=-1, help='number of views (default: -1, all views)')
    parser.add_argument('--resize', type=int, default=-1, help='resize the input images to (default: -1, no resize)')
    parser.add_argument('--tigre_algo', type=str, default='fdk', choices=['fdk', 'sart', 'sart_tv', 'os_asd_pocs'])
    parser.add_argument('--tigre_iters', type=int, default=20)
    args = parser.parse_args()
    main(args)
