
object_name=diamond

/home/mscfanuc/anaconda3/envs/dttdnet/bin/python ./interactive_mask.py --object $object_name
/home/mscfanuc/anaconda3/envs/dust3r/bin/python ./dust3r_insertion.py --object $object_name