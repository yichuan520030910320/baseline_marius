def extract_accuracies_from_file(file_name):
    accuracies = []
    with open(file_name, 'r') as file:
        lines = file.readlines()
    
    i = 0
    val_accs = []
    test_accs = []
    while i < len(lines):
        line = lines[i].strip()
        # Find the line ending with 'Evaluating validation set'
        if line.endswith('Evaluating validation set'):
            i += 1  # Move to the next line
            # Now, look for the next two occurrences of lines that begin with 'Accuracy:'
            accuracy_count = 0
            while i < len(lines) and accuracy_count < 2:
                if lines[i].strip().startswith('Accuracy:'):
                    # Extract and store the accuracy percentage
                    accuracy = lines[i].split(':')[1].strip().replace('%', '')
                    try:
                        accuracies.append(round(float(accuracy), 2))
                        accuracy_count += 1
                        if accuracy_count==1:
                            val_accs.append(round(float(accuracy), 2))
                        if accuracy_count==2:
                            test_accs.append(round(float(accuracy), 2))
                            
                    except ValueError:
                        pass
                i += 1
        else:
            i += 1
    max_val_acc=max(val_accs)
    ## corresponding test acc
    max_test_acc=test_accs[val_accs.index(max_val_acc)]
    max_indices=val_accs.index(max_val_acc)
    print('max_indices:',max_indices)
    print(f"Max val acc: {max_val_acc}, corresponding test acc: {max_test_acc}")
    return accuracies
import argparse
import os
parser = argparse.ArgumentParser(description='Extract accuracies from log files')
parser.add_argument('--log_dir', type=str, help='Directory containing log files',default='/nvme2n1/marius/datasets/init_mag_new8192/mag_324.log')
args=parser.parse_args()
extract_accuracies_from_file(args.log_dir)