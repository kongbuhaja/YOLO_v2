import tensorflow as tf

def apply(img, grid):
    for augmentation_method in [random_brightness, random_contrast, random_hue, random_saturation, flip_horizontally]:
        img, grid = randomly_apply_operation(augmentation_method, img, grid)
    return img, grid

def get_random_bool():
    return tf.greater(tf.random.uniform((), dtype=tf.float32), 0.5)

def randomly_apply_operation(operation, img, grid, *args):
    return tf.cond(
        get_random_bool(),
        lambda: operation(img, grid, *args),
        lambda: (img, grid)        
    )

def random_brightness(img, grid, max_delta=0.12):
    return tf.image.random_brightness(img, max_delta), grid

def random_contrast(img, grid, lower=0.5, upper=1.5):
    return tf.image.random_contrast(img, lower, upper), grid

def random_hue(img, grid, max_delta=0.08):
    return tf.image.random_hue(img, max_delta), grid

def random_saturation(img, grid, lower=0.5, upper=1.5):
    return tf.image.random_saturation(img, lower, upper), grid

def flip_horizontally(img, grid, out_size=13):
    flipped_img = tf.image.flip_left_right(img)

    flipped_grid = tf.concat([out_size - grid[..., :1], grid[..., 1:]],-1)
    return flipped_img, flipped_grid