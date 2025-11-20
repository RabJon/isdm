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


def balance(masks_or_mask_file_paths, num_samples, num_classes, ejection_classes = [0], random_seed = 73):
    
    def uniques_counts_wo_ejections(array, ejection_classes):
        uniques, unique_counts = np.unique(array, return_counts=True)
        bool_index = np.isin(uniques, ejection_classes, invert=True)
        return uniques[bool_index], unique_counts[bool_index]

    
    rng = np.random.default_rng(random_seed)
    masks_or_mask_file_paths = np.array(masks_or_mask_file_paths)
    num_real_masks = len(masks_or_mask_file_paths)


    provided_as_masks = isinstance(masks_or_mask_file_paths[0], np.ndarray)
    if provided_as_masks:
        Y = masks_or_mask_file_paths
        Y[Y >= num_classes] = 0 #setting to zero, because we can't generate regions the model was not trained for
    else:
        Y = None
        for i in range(num_real_masks):
            mask = Image.open(masks_or_mask_file_paths[i]).convert('L')
            mask[mask >= num_classes] = 0 #setting to zero, because we can't generate regions the model was not trained for
            if i == 0:
                Y = np.zeros((num_real_masks, mask.shape[0], mask.shape[1]), dtype=np.uint8)

            Y[i] = np.array(mask)


    uniques, unique_counts = uniques_counts_wo_ejections(Y, ejection_classes) 
    class_weights = np.sum(unique_counts) / unique_counts.astype(np.float32)
    class_weights_dict = {c : w for c, w in zip(uniques, class_weights)}
    sample_weights = np.zeros((len(Y),), dtype=np.float32)

    for i in range(len(Y)):
        uniques, unique_counts = uniques_counts_wo_ejections(Y[i], ejection_classes) 
        
        if len(uniques) == 0:
            sample_weights[i] = 0.0
            continue
        
        most_frequent_class = uniques[np.argmax(unique_counts)]
        #assign the class weight of the most frequent class in the mask (heurisitcally proved to work well)
        sample_weights[i] = class_weights_dict[most_frequent_class] 
        

    balance_indices = rng.choice(np.arange(len(Y)), size = num_samples, p = sample_weights / np.sum(sample_weights))
    
    return balance_indices.astype(np.uint32)


    
    # shape = Y.shape
    # reshaped_masks = Y.squeeze().reshape( (shape[0], shape[1] * shape[2]) )
    # indices_non_zero = ~np.all(reshaped_masks==0, axis=1) #get indices of not sound only images

    # mask_classes = np.unique(Y[np.nonzero(Y)]).astype(np.uint8)
    # if based_on_generated_data_only:
    #     #select 1/10 of the required data randomly as a starting point
    #     available_indices = np.arange(len(Y))[indices_non_zero]
    #     random_indices = rng.choice(available_indices, size=num_samples//8, replace=False)
    #     class_counts_orig = get_class_counts_frequencies(Y[random_indices], num_defects=num_defects)[0]
    #     balance_mask_indices = random_indices.tolist()
    #     num_generated_masks = len(random_indices)
    # else:
    #     class_counts_orig = get_class_counts_frequencies(Y, num_defects=num_defects)[0]
    
    # was_balanced = False
    # if not based_on_generated_data_only:
    #     balance_mask_indices = []
    #     num_generated_masks = 0
    # class_counts = list(class_counts_orig)
    # while num_generated_masks < num_samples:
    #     max_c = np.max(class_counts)

    #     completed_classes = [
    #         cs+1 for cs, count in enumerate(class_counts) if count*(1.0 + max_class_dif) >= max_c]
        
    #     if len(completed_classes) == num_defects:
    #         print("All classes balanced with maximal class difference of", max_class_dif, 
    #               "Reducing max class difference to", 0.9*max_class_dif)
    #         was_balanced = True
    #         max_class_dif = 0.9 * max_class_dif
    #         completed_classes = [
    #             cs+1 for cs, count in enumerate(class_counts) if count*(1.0 + max_class_dif) >= max_c]

    #     indices = np.array(indices_non_zero)
    #     class_frequencies = np.array(class_counts) / np.sum(class_counts)
    #     indices_sort_keys = np.ones((len(indices),))
    #     for c in mask_classes:
    #         no_class_c_included = np.all(reshaped_masks != c , axis=1)
    #         if (class_counts[c-1] * (1.0 + max_class_dif) >= max_c):
    #             indices *= no_class_c_included #exclude already balanced classes
    #         else:
    #             class_c_included = ~no_class_c_included
    #             if class_frequencies[c-1] < indices_sort_keys[class_c_included][0]:
    #                 indices_sort_keys[class_c_included] = class_frequencies[c-1]
            
    #     indices_sort_keys = indices_sort_keys[indices]
    #     indices = np.flatnonzero(indices)
        
    #     #shuffle indices (and keys) to avoid that always the same indices get choosen first
    #     #even if already the sorting can permute indics of same importance, with this we add
    #     #addtional randomness
    #     rand_perm = rng.permutation(len(indices))
    #     indices = indices[rand_perm]
    #     indices_sort_keys = indices_sort_keys[rand_perm]
    #     indices = indices[indices_sort_keys.argsort()] #importance sorting for the indices 

    #     if len(indices) == 0:
    #         print("No more masks that contribute to balancing!")
    #         break
        
    #     #shuffle the indices randomly to avoid that always the same masks are selected
    #     #rng.shuffle(indices) replaced by sorting

    #     if was_balanced and (max_class_dif < 0.1):
    #         #Dataset was already balanced and max_class_dif is below 0.1.
    #         #Therefore, it is required to add masks more carefully.
    #         #to avoid destroying the balance too much.
    #         num_remaining_masks = num_samples - num_generated_masks
    #         num_allowed = (num_remaining_masks//8)
    #         num_allowed = 2 if num_allowed < 2 else num_allowed
    #         indices = indices[:num_allowed]

    #     for mask, idx in zip(Y[indices], indices):
    #         balance_mask_indices += [idx]
    #         num_generated_masks += 1

    #         class_counts = np.array(update_class_counts(class_counts, mask))
    #         upscaled_counts = class_counts * (1.0 + max_class_dif)
            
    #         num_completed_classes_prev = len(completed_classes)
    #         num_completed_classes_now = np.count_nonzero(upscaled_counts >= max_c)
    #         if (num_completed_classes_now > num_completed_classes_prev):
    #             print(
    #                 (num_completed_classes_now - num_completed_classes_prev),
    #                 f"new classes balanced. Sampled {num_generated_masks} masks.", 
    #                 f"Class counts are {class_counts}")
    #             if num_generated_masks < num_samples:
    #                 print("Starting new iteration... ")
    #             break


    
    # if num_generated_masks < num_samples:
        
    #     num_remaining_masks = (num_samples - num_generated_masks)
    #     print(f"Complete balancing was not possible! Stopping balancing after simulating {num_generated_masks} generated masks.")
    #     print(f"The remaining {num_remaining_masks} are resampled from the 'balanced' ones!")
    #     #balancing completely was not possible, but the masks in balance_mask_paths
    #     #are as balanced as possible to lets resample from there until we have num_samples masks
    #     additional_mask_indices = rng.choice(balance_mask_indices, size = num_remaining_masks)
    #     balance_mask_indices += additional_mask_indices.tolist()
        
        
    
    
    # return np.array(balance_mask_indices[:num_samples], dtype=np.uint32)