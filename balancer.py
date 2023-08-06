import numpy as np
from torch.utils.data import Dataset
from PIL import Image




def get_class_counts_frequencies(
    y_masks, num_defects, class_id = None, 
    with_ejection_class = False):
    """Returns pixelwise counts of each defect class in y_masks and their 
    relative frequencies. Per default the ejection class (class 0) is omitted.

    Args:
            y_masks (ndarray): The mask of class ids, can either be a single mask
                or a array of masks
            num_defects (int, optional): The number of defects to consider.
            class_id (int, optional): The id of the class in the mask for 
                which to return the results. Defaults to None. If None it 
                is returned for all classes.
            with_ejection_class (bool, optional): Flag indicating whether to 
                include the ejection class. Defaults to False.

    Raises:
            ValueError: If class_id is set to 0 and with_ejection_class to False

    Returns:
            tuple: pixel counts of each class and their percentual frequencies
    """
        
    num_classes = num_defects if not with_ejection_class else num_defects + 1

    if class_id is None:
        class_counts = [0] * num_classes

        if with_ejection_class:
                sum_counts = y_masks.size
                class_counts[0] = sum_counts - np.count_nonzero(y_masks)

        for i in range(1,num_defects+1):
            if with_ejection_class:
                class_counts[i] = np.count_nonzero(y_masks == i)
            else:
                class_counts[i-1] = np.count_nonzero(y_masks == i)
        
        if not with_ejection_class:
            sum_counts = sum(class_counts)
        
        if sum_counts > 0:
            class_freq = [float(count)/float(sum_counts) for count in class_counts]
            return class_counts, class_freq
        else:
            return class_counts, class_counts
    
    else:
        if with_ejection_class:
            total_class_count = y_masks.size
            count = np.count_nonzero(y_masks == class_id) if class_id != 0\
                else total_class_count - np.count_nonzero(y_masks)
        else:
            if class_id == 0:
                raise ValueError("class_id 0 is not valid when calling without exception class!")
            total_class_count = np.count_nonzero(y_masks)
            count = np.count_nonzero(y_masks == class_id)
            
        if total_class_count > 0:
            return count, float(count)/total_class_count
        else:
            return 0, 0.0


def balance(
        mask_file_paths, 
        num_samples,
        num_classes,
        height = 256,
        width = 256,
        max_class_dif = 0.1,
        based_on_generated_data_only = False, 
        based_on_pixel_count=True, 
        ejection_classes = [0], 
        use_mask_augmentation=False):


    if use_mask_augmentation:
        raise NotImplementedError("Mask augmentation is not implemented yet!")
    if ejection_classes != [0]:
        raise NotImplementedError("Ejection classes are not implemented yet!")
    if based_on_generated_data_only:
        raise NotImplementedError("Based on generated data only is not implemented yet!")
    if not based_on_pixel_count:
        raise NotImplementedError("Based on object count is not implemented yet!")
    
    
    mask_file_paths = np.array(mask_file_paths)
    num_real_masks = len(mask_file_paths)
    num_defects = num_classes - 1

    
    Y = np.zeros((num_real_masks, height, width), dtype=np.uint8)
    for i in range(num_real_masks):
        mask = Image.open(mask_file_paths[i]).convert('L').resize(width, height, resample=Image.NEAREST)
        if num_classes == 2:
            mask[mask == 255] = 1
        Y[i] = np.array(mask)
    
    
    mask_classes = np.unique(Y[np.nonzero(Y)]).astype(np.uint8)
    class_counts_orig = get_class_counts_frequencies(Y, num_defects=num_defects)[0]
    
    shape = Y.shape
    reshaped_masks = Y.squeeze().reshape( (shape[0], shape[1] * shape[2]) )
    indices_non_zero = ~np.all(reshaped_masks==0, axis=1) #get indices of not sound only images

    update_class_counts = lambda counts, new_masks:\
        [
            y+x for x,y in zip(get_class_counts_frequencies(new_masks, num_defects=num_defects)[0], counts) 
        ] 


    balance_mask_indices = []
    class_counts = list(class_counts_orig)
    num_generated_masks = 0
    while num_generated_masks < num_samples:
        max_c = np.max(class_counts)

        completed_classes = [
            cs+1 for cs, count in enumerate(class_counts) if count*(1.0 + max_class_dif) >= max_c]
        
        if len(completed_classes) == num_defects:
            print("All classes balanced with maximal class difference of", max_class_dif, 
                  "Reducing max class difference to", 0.9*max_class_dif)
            max_class_dif = 0.9 * max_class_dif
            completed_classes = [
                cs+1 for cs, count in enumerate(class_counts) if count*(1.0 + max_class_dif) >= max_c]

        indices = np.array(indices_non_zero)
        for c in mask_classes:
            if (class_counts[c-1] * (1.0 + max_class_dif) >= max_c):
                indices *= np.all(reshaped_masks != c , axis=1) #exclude already balanced classes
        indices = np.flatnonzero(indices)

        if len(indices) == 0:
            print("No more masks that contribute to balancing!")
            break
        
        #TODO: instead of random, sort indices by class counts that appear infrequently
        np.random.seed(73) #wrong because init in each loop, but never same indices so ok
        np.random.shuffle(indices)


        for mask, idx in zip(Y[indices], indices):
            balance_mask_indices += [idx]
            num_generated_masks += 1

            class_counts = np.array(update_class_counts(class_counts, mask))
            upscaled_counts = class_counts * (1.0 + max_class_dif)
            
            num_completed_classes_prev = len(completed_classes)
            num_completed_classes_now = np.count_nonzero(upscaled_counts >= max_c)
            if num_completed_classes_now > num_completed_classes_prev:
                print(
                    (num_completed_classes_now - num_completed_classes_prev),
                    "new classes balanced. Class counts are ", class_counts, ". Starting new iteration... ")
                break

    
    if num_generated_masks < num_samples:
        #balancing completely was not possible, but the masks in balance_mask_paths
        #are as balanced as possible to lets resample from there until we have num_samples masks
        additional_mask_indices = np.random.choice(balance_mask_indices, size = (num_samples - num_generated_masks))
        balance_mask_indices += additional_mask_indices.tolist()
        
        
    
    
    return np.array(balance_mask_indices[:num_samples], dtype=np.uint32)