import numbers
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import random


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomCrop(object):
    """
    Take a random crop from the image.
    First the image or crop size may need to be adjusted if the incoming image
    is too small...
    If the image is smaller than the crop, then:
         the image is padded up to the size of the crop
         unless 'nopad', in which case the crop size is shrunk to fit the image
    A random crop is taken such that the crop fits within the image.
    If a centroid is passed in, the crop must intersect the centroid.
    """
    def __init__(self, size=512, ignore_index=12, nopad=True):

        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.ignore_index = ignore_index
        self.nopad = nopad
        self.pad_color = (0, 0, 0)

    def __call__(self, img, mask, centroid=None):
        assert img.size == mask.size
        w, h = img.size
        # ASSUME H, W
        th, tw = self.size
        if w == tw and h == th:
            return img, mask

        if self.nopad:
            if th > h or tw > w:
                # Instead of padding, adjust crop size to the shorter edge of image.
                shorter_side = min(w, h)
                th, tw = shorter_side, shorter_side
        else:
            # Check if we need to pad img to fit for crop_size.
            if th > h:
                pad_h = (th - h) // 2 + 1
            else:
                pad_h = 0
            if tw > w:
                pad_w = (tw - w) // 2 + 1
            else:
                pad_w = 0
            border = (pad_w, pad_h, pad_w, pad_h)
            if pad_h or pad_w:
                img = ImageOps.expand(img, border=border, fill=self.pad_color)
                mask = ImageOps.expand(mask, border=border, fill=self.ignore_index)
                w, h = img.size

        if centroid is not None:
            # Need to insure that centroid is covered by crop and that crop
            # sits fully within the image
            c_x, c_y = centroid
            max_x = w - tw
            max_y = h - th
            x1 = random.randint(c_x - tw, c_x)
            x1 = min(max_x, max(0, x1))
            y1 = random.randint(c_y - th, c_y)
            y1 = min(max_y, max(0, y1))
        else:
            if w == tw:
                x1 = 0
            else:
                x1 = random.randint(0, w - tw)
            if h == th:
                y1 = 0
            else:
                y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class PadImage(object):
    def __init__(self, size=(512, 512), ignore_index=0):
        self.size = size
        self.ignore_index = ignore_index

    def __call__(self, img, mask):
        assert img.size == mask.size
        th, tw = self.size, self.size

        w, h = img.size

        if w > tw or h > th:
            wpercent = (tw / float(w))
            target_h = int((float(img.size[1]) * float(wpercent)))
            img, mask = img.resize((tw, target_h), Image.BICUBIC), mask.resize((tw, target_h), Image.NEAREST)

        w, h = img.size
        img = ImageOps.expand(img, border=(0, 0, tw - w, th - h), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, tw - w, th - h), fill=self.ignore_index)

        return img, mask


class RandomHorizontalFlip(object):

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if mask is not None:
            if random.random() < self.prob:
                return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(
                    Image.FLIP_LEFT_RIGHT)
            else:
                return img, mask
        else:
            if random.random() < self.prob:
                return img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                return img


class RandomVerticalFlip(object):
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if mask is not None:
            if random.random() < self.prob:
                return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(
                    Image.FLIP_TOP_BOTTOM)
            else:
                return img, mask
        else:
            if random.random() < self.prob:
                return img.transpose(Image.FLIP_TOP_BOTTOM)
            else:
                return img


class Resize(object):
    def __init__(self, size: tuple = (512,  512)):
        self.size = size  # size: (h, w)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return img.resize(self.size, Image.BICUBIC), mask.resize(self.size, Image.NEAREST)


class RandomScale(object):
    def __init__(self, scale_list=[0.75, 1.0, 1.25], mode='value'):
        self.scale_list = scale_list
        self.mode = mode

    def __call__(self, img, mask):
        oh, ow = img.size
        scale_amt = 1.0
        if self.mode == 'value':
            scale_amt = np.random.choice(self.scale_list, 1)
        elif self.mode == 'range':
            scale_amt = random.uniform(self.scale_list[0], self.scale_list[-1])
        h = int(scale_amt * oh)
        w = int(scale_amt * ow)
        return img.resize((w, h), Image.BICUBIC), mask.resize((w, h), Image.NEAREST)


class ColorJitter(object):
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5):
        if not brightness is None and brightness>0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast>0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation>0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, img, mask=None):
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        img = ImageEnhance.Brightness(img).enhance(r_brightness)
        img = ImageEnhance.Contrast(img).enhance(r_contrast)
        img = ImageEnhance.Color(img).enhance(r_saturation)
        if mask is None:
            return img
        else:
            return img, mask


class SmartCropV1(object):
    """
类的构造函数__init__接受以下参数：

crop_size：裁剪后的目标大小，默认为512。
max_ratio：裁剪后的图像中最大类别（除去ignore_index）的像素数与总像素数之比的阈值。如果超过该阈值，将继续进行裁剪操作，直到满足条件。默认值为0.75。
ignore_index：用于忽略的掩膜像素值的索引，默认为12。
nopad：是否对图像进行填充的布尔值，默认为False。
在类的初始化过程中，通过调用RandomCrop类创建了一个裁剪操作self.crop，传入了裁剪大小和忽略索引等参数。

类中定义了__call__方法，用于实现类的可调用行为。该方法接受图像img和掩膜mask作为输入，并执行智能裁剪操作。

智能裁剪的过程如下：

使用self.crop对输入的图像和掩膜进行裁剪操作，返回裁剪后的图像和掩膜。
统计裁剪后掩膜中每个类别的像素数，并计算除去ignore_index后最大类别的像素数与总像素数之比。
如果存在多个类别（除去ignore_index）且最大类别的像素数比例未超过阈值（max_ratio），则跳出循环，结束裁剪过程。
如果裁剪次数超过10次，或者无法满足条件（即不存在多个类别或最大类别比例超过阈值），则跳出循环，结束裁剪过程。
最后，返回裁剪后的图像和掩膜作为方法的输出。

通过使用SmartCropV1类，可以实现根据指定条件智能裁剪图像和对应的掩膜，以用于数据增强或其他任务的需要。
    """
    def __init__(self, crop_size=512,
                 max_ratio=0.75,
                 ignore_index=12, nopad=False):
        self.crop_size = crop_size
        self.max_ratio = max_ratio
        self.ignore_index = ignore_index
        self.crop = RandomCrop(crop_size, ignore_index=ignore_index, nopad=nopad)

    def __call__(self, img, mask):
        assert img.size == mask.size
        count = 0
        while True:
            img_crop, mask_crop = self.crop(img.copy(), mask.copy())
            count += 1
            labels, cnt = np.unique(np.array(mask_crop), return_counts=True)
            cnt = cnt[labels != self.ignore_index]
            if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.max_ratio:
                break
            if count > 10:
                break

        return img_crop, mask_crop


class SmartCropV2(object):
    def __init__(self, crop_size=512, num_classes=13,
                 class_interest=[2, 3],
                 class_ratio=[0.1, 0.25],
                 max_ratio=0.75,
                 ignore_index=12, nopad=True):
        self.crop_size = crop_size
        self.num_classes = num_classes
        self.class_interest = class_interest
        self.class_ratio = class_ratio
        self.max_ratio = max_ratio
        self.ignore_index = ignore_index
        self.crop = RandomCrop(crop_size, ignore_index=ignore_index, nopad=nopad)

    def __call__(self, img, mask):
        assert img.size == mask.size
        count = 0
        while True:
            img_crop, mask_crop = self.crop(img.copy(), mask.copy())
            count += 1
            bins = np.array(range(self.num_classes + 1))
            class_pixel_counts, _ = np.histogram(np.array(mask_crop), bins=bins)
            cf = class_pixel_counts / (self.crop_size * self.crop_size)
            cf = np.array(cf)
            for c, f in zip(self.class_interest, self.class_ratio):
                if cf[c] > f:
                    break
            if np.max(cf) < 0.75 and np.argmax(cf) != self.ignore_index:
                break
            if count > 10:
                break

        return img_crop, mask_crop