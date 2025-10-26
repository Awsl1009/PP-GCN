import torch
import clip

part_description_map = []
with open('/root/DGait_CTRGCN/text/dgait_partdescription_openai.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split(';')
        part_description_map.append(temp_list)


def text_prompt_part_description_4part_clip():
    print("Use text prompt part description for clip")
    text_dict = {}
    num_text_aug = 5

    for ii in range(num_text_aug):
        if ii == 0:
            # text_dict[0]:label0(normal) and label1(depression)
            text_dict[ii] = torch.cat([clip.tokenize((pasta_list[ii])) for pasta_list in part_description_map])
        elif ii == 1:
            # text_dict[1]:label0 + head and label1 + head
            text_dict[ii] = torch.cat(
                [clip.tokenize((','.join(pasta_list[0:2]))) for pasta_list in part_description_map])
        elif ii == 2:
            # text_dict[2]:label0 + hand + arm and label1 + hand + arm
            text_dict[ii] = torch.cat(
                [clip.tokenize((pasta_list[0] + ','.join(pasta_list[2:4]))) for pasta_list in part_description_map])
        elif ii == 3:
            # text_dict[3]:label0 + hip and label1 + hip
            text_dict[ii] = torch.cat(
                [clip.tokenize((pasta_list[0] + ',' + pasta_list[4])) for pasta_list in part_description_map])
        else:
            # text_dict[4]:label0 + leg + foot and label1 + leg + foot
            text_dict[ii] = torch.cat(
                [clip.tokenize((pasta_list[0] + ',' + ','.join(pasta_list[5:]))) for pasta_list in
                 part_description_map])

    classes = torch.cat([v for k, v in text_dict.items()])

    return classes, num_text_aug, text_dict


def text_prompt_overall_description_clip():
    print("Use overall text prompt for clip")
    text_dict = {}
    num_text_aug = 2

    with open('/root/DGait_CTRGCN/text/dgait_overalldescription_openai.txt') as infile:
        lines = infile.readlines()
        for ii in range(num_text_aug):
            line = lines[ii]
            text_dict[ii] = clip.tokenize(line)

    classes = torch.cat([v for k, v in text_dict.items()])
    return classes, num_text_aug, text_dict
