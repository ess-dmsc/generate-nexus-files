import time
from datetime import datetime, timedelta

from file_writer_control import JobHandler
from file_writer_control import WorkerCommandChannel
from file_writer_control import WriteJob
from os import path

if __name__ == '__main__':
    kafka_host = 'localhost:9092'
    command_channel = WorkerCommandChannel(f'{kafka_host}/Loki_writerCommand')
    job_handler = JobHandler(worker_finder=command_channel)
    nexus_structure_file = path.join('..', 'NXtomo.json')
    # nexus_structure_file = path.join('..', 'nexus_config.json')
    start_time = datetime.now()
    with open(nexus_structure_file, 'r') as f:
        nexus_structure = f.read()
    write_job = WriteJob(
        nexus_structure,
        "{0:%Y}-{0:%m}-{0:%d}_{0:%H}{0:%M}.nxs".format(start_time),
        kafka_host,
        start_time,
    )
    # print(nexus_structure)

    print('Starting write job')
    start_handler = job_handler.start_job(write_job)
    while not start_handler.is_done():
        time.sleep(1)
    stop_time = start_time + timedelta(seconds=230)
    stop_handler = job_handler.set_stop_time(stop_time)
    while not stop_handler.is_done():
        time.sleep(1)
    while not job_handler.is_done():
        time.sleep(1)
    print('Write job is done')
