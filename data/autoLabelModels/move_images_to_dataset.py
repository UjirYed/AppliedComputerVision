from claude_autoLabel import move_images_to_dataset

image_link_file_path = "../imageScraping/LLMBrowser/image_urls/outputaa"
label_file_path = "label_file_processed.txt"
dataset_path = "testdataset"
acceptable_labels = ['0','1','2','3','4','5', 'N/A']

move_images_to_dataset(image_link_file_path, label_file_path, dataset_path, acceptable_labels=acceptable_labels)


image_link_file_path = "../imageScraping/LLMBrowser/image_urls/outputab"
label_file_path = "label_file2_processed.txt"
move_images_to_dataset(image_link_file_path, label_file_path, dataset_path, acceptable_labels=acceptable_labels, offset=3000)