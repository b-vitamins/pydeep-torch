"""
Provides a thread/script pooling mechanism based on ssh + screen.

:Version:
    1.1.0

:Date:
    19.03.2017

:Author:
    Jan Melchior

:Contact:
    JanMelchior@gmx.de

:License:

    Copyright (C) 2017 Jan Melchior

    This file is part of the Python library PyDeep.

    PyDeep is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import paramiko
from encryptedpickle import encryptedpickle
import cPickle
import datetime
import torch
import socket
import pickle


def _torch_div_and_mul(load_val, cpu_count):
    """
    Use torch for (100.0 * load_val / cpu_count). Returns a float.
    """
    load_t = torch.tensor(float(load_val), dtype=torch.float64)
    cpu_t = torch.tensor(float(cpu_count), dtype=torch.float64)
    if cpu_t.item() == 0.0:
        return 0.0
    return float(100.0 * (load_t / cpu_t).item())


def _torch_sub(cpu_count, load_val):
    """
    Torch-based sub: (cpu_count - load_val). Returns a float.
    """
    c_t = torch.tensor(float(cpu_count), dtype=torch.float64)
    l_t = torch.tensor(float(load_val), dtype=torch.float64)
    return float((c_t - l_t).item())


class SSHConnection(object):
    """Handles a SSH connection."""

    def __init__(self, hostname, username, password, max_cpus_usage=2):
        """Constructor takes hostname, username, password.

        :param hostname: Hostname or address of host.
        :type hostname: string

        :param username: SSH username.
        :type username: string

        :param password: SSH password.
        :type password: string

        :param max_cpus_usage: Maximal number of cores to be used
        :type max_cpus_usage: int
        """
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.username = username
        self.password = password
        self.hostname = hostname
        self.architecture = "Unknown"
        self.cpu_count = 0
        self.cpu_speed = 0
        self.max_cpus_usage = max_cpus_usage
        self._free_cpus_last_request = 0.0
        self.memory_size = 0
        self.raw_cpu_info = {}
        self.raw_memory_info = {}
        self.is_connected = False

    def encrypt(self, password):
        """Encrypts the connection object.

        :param password: Encryption password
        :type password: string

        :return: Encrypted object
        :rtype: object
        """
        passphrases = {0: password}
        encoder = encryptedpickle.EncryptedPickle(
            signature_passphrases=passphrases, encryption_passphrases=passphrases
        )
        return encoder.seal(self)

    @classmethod
    def decrypt(cls, connection, password):
        """Decrypts a connection object and returns it

        :param connection: SSHConnection to be decrypted
        :type connection: string

        :param password: Encryption password
        :type password: string

        :return: Decrypted object
        :rtype: SSHConnection
        """
        passphrases = {0: password}
        encoder = encryptedpickle.EncryptedPickle(
            signature_passphrases=passphrases, encryption_passphrases=passphrases
        )
        return encoder.unseal(connection)

    def connect(self):
        """Connects to the server.

        :return: True if the connection was successful, False otherwise
        :rtype: bool
        """
        self.disconnect()
        try:
            self.client.connect(
                hostname=self.hostname,
                username=self.username,
                password=self.password,
            )
            self.is_connected = True
        except (
            paramiko.BadHostKeyException,
            paramiko.AuthenticationException,
            paramiko.SSHException,
            socket.error,
        ) as e:
            self.is_connected = False
            print(f"Connection failed: {e}")
        return self.is_connected

    def disconnect(self):
        """Disconnects from the server."""
        self.client.close()
        self.is_connected = False

    def execute_command(self, command):
        """Executes a command on the server and returns stdin, stdout, and stderr

        :param command: Command to be executed.
        :type command: string

        :return: (stdin, stdout, stderr) or (None, None, None) if not connected
        :rtype: tuple
        """
        if not self.is_connected:
            self.connect()
        if self.is_connected:
            return self.client.exec_command(command)
        else:
            return None, None, None

    def execute_command_in_screen(self, command):
        """Executes a command in a screen on the server which is automatically detached.

        :param command: Command to be executed.
        :type command: string

        :return: (stdin, stdout, stderr) or (None, None, None)
        :rtype: tuple
        """
        return self.execute_command(command="screen -d -m " + command)

    def renice_processes(self, value):
        """Renices all processes.

        :param value: The New nice value -40 ... 20
        :type value: int or string

        :return: (stdin, stdout, stderr)
        :rtype: tuple
        """
        return self.execute_command("renice " + str(value) + " -u " + self.username)

    def kill_all_processes(self):
        """Kills all processes.

        :return: (stdin, stdout, stderr)
        :rtype: tuple
        """
        return self.execute_command("killall -u " + self.username)

    def kill_all_screen_processes(self):
        """Kills all screen processes.

        :return: (stdin, stdout, stderr)
        :rtype: tuple
        """
        return self.execute_command("killall -15 screen")

    def get_server_info(self):
        """Get the server info (CPU count, memory size, etc.) and store them internally.

        :return: "online" if connected, "offline" otherwise
        :rtype: string
        """
        if not self.is_connected:
            self.connect()
        if self.is_connected:
            # Get CPU info
            _, stdout, _ = self.execute_command("lscpu")
            stdout_lines = stdout.readlines()
            for item in stdout_lines:
                kvp = item.split(":")
                if len(kvp) < 2:
                    continue
                self.raw_cpu_info[kvp[0]] = kvp[1].replace(" ", "")

            if "CPU(s)" in self.raw_cpu_info:
                self.cpu_count = int(self.raw_cpu_info["CPU(s)"])
            if "CPU op-mode(s)" in self.raw_cpu_info:
                self.architecture = self.raw_cpu_info["CPU op-mode(s)"]
            if "Thread(s)percore" in self.raw_cpu_info:
                self.cpu_speed = int(self.raw_cpu_info["Thread(s)percore"])

            # Get memory info
            _, stdout, _ = self.execute_command("free -m")
            lines = stdout.readlines()
            if len(lines) > 1:
                keys = lines[0].split()
                values = lines[1].split()
                self.raw_memory_info = {}
                for i in range(len(keys)):
                    if (i + 1) < len(values):
                        self.raw_memory_info[keys[i]] = values[i + 1]
                if "total" in self.raw_memory_info:
                    self.memory_size = int(self.raw_memory_info["total"])
            return "online"
        return "offline"

    def get_server_load(self):
        """Get the current CPU and memory usage of the server.

        :return: (load1, load5, load15, used_memMB) or (None,None,None,None)
        :rtype: tuple
        """
        if not self.is_connected:
            self.connect()
        if self.is_connected:
            if self.cpu_count == 0:
                self.get_server_info()

            _, stdout, _ = self.execute_command("cat /proc/loadavg")
            load_line = stdout.readlines()[0].split()
            cpu_load = [float(load_line[0]), float(load_line[1]), float(load_line[2])]

            _, stdout, _ = self.execute_command("free -m")
            lines = stdout.readlines()
            mem_used = 0
            if len(lines) > 1:
                mem_used = int(lines[1].split()[2])
            self._free_cpus_last_request = self.cpu_count - cpu_load[0]
            return cpu_load[0], cpu_load[1], cpu_load[2], mem_used
        return None, None, None, None

    def get_number_users_processes(self):
        """Gets number of processes of the user on the server.

        :return: number of processes or None
        :rtype: int or None
        """
        res = self.execute_command("ps aux | grep -c " + self.username)[1]
        if res is None:
            return None
        return int(res.readlines()[0])

    def get_number_users_screens(self):
        """Gets number of users' screens on the server.

        :return: number of users' screens or None
        :rtype: int or None
        """
        res1 = self.execute_command("screen -ls | grep -c Attached")[1]
        if res1 is None:
            return None
        res2 = self.execute_command("screen -ls | grep -c Detached")[1]
        return int(res1.readlines()[0]) + int(res2.readlines()[0])


class SSHJob(object):
    """Handles a SSH JOB."""

    def __init__(self, command, num_threads=1, nice=19):
        """
        :param command: Command to be executed.
        :type command: string

        :param num_threads: Number of threads the job needs.
        :type num_threads: int

        :param nice: Nice value for this job.
        :type nice: int
        """
        self.command = command
        self.num_threads = num_threads
        self.nice = nice


class SSHPool(object):
    """Handles a pool of servers and allows to distribute jobs over the pool."""

    def __init__(self, servers):
        """Constructor takes a list of SSHConnections.

        :param servers: List of SSHConnections.
        :type servers: list
        """
        self.servers = servers
        self.log = []

    def save_server(self, path, password):
        """Saves the encrypted server list to path.

        :param path: Path and filename
        :type path: string

        :param password: Encryption password
        :type password: string
        """
        encrypted_server_list = []
        for s in self.servers:
            encrypted_server_list.append(s.encrypt(password))
        try:
            with open(path, "w") as f:
                cPickle.dump(encrypted_server_list, f)
            self.log.append(str(datetime.datetime.now()) + " Server saved to " + path)
        except (OSError, IOError, pickle.PickleError) as e:
            raise Exception("-> File writing Error: " + str(e))

    def load_server(self, path, password, append=True):
        """Loads an encrypted server list from disk.

        :param path: Path and filename.
        :type path: string

        :param password: Encryption password.
        :type password: string

        :param append: If true, servers are appended; if false, the server list is replaced.
        :type append: bool
        """
        try:
            with open(path, "r") as f:
                encrypted_server_list = cPickle.load(f)
        except (OSError, IOError, pickle.PickleError) as e:
            raise Exception("-> File reading Error: " + str(e))

        if not append:
            self.servers.clear()
        try:
            for enc_obj in encrypted_server_list:
                self.servers.append(enc_obj.decrypt(password))
            self.log.append(
                str(datetime.datetime.now()) + " Server loaded from " + path
            )
        except encryptedpickle.EncryptedPickleError:
            raise Exception("Wrong password or decryption error!")

    def execute_command(self, host, command):
        """Executes a command on a given server.

        :param host: Hostname or SSHConnection object
        :type host: string or SSHConnection

        :param command: Command to be executed
        :type command: string

        :return: result from exec_command or "offline"
        :rtype: (stdin, stdout, stderr) or "offline"
        """
        if isinstance(host, SSHConnection):
            s = host
        else:
            s = self.servers[host]
        output = "offline"
        if s.connect():
            output = s.execute_command(command)
            self.log.append(
                str(datetime.datetime.now())
                + " Command "
                + command
                + " executed on "
                + s.hostname
            )
        s.disconnect()
        return output

    def execute_command_in_screen(self, host, command):
        """Executes a command in a screen on a given server.

        :param host: Hostname or SSHConnection object
        :type host: string or SSHConnection

        :param command: Command to be executed
        :type command: string

        :return: result from exec_command_in_screen or "offline"
        :rtype: (stdin, stdout, stderr) or "offline"
        """
        if isinstance(host, SSHConnection):
            s = host
        else:
            s = self.servers[host]
        output = "offline"
        if s.connect():
            output = s.execute_command_in_screen(command)
            self.log.append(
                str(datetime.datetime.now())
                + " Command in screen "
                + command
                + " executed on "
                + s.hostname
            )
        s.disconnect()
        return output

    def broadcast_command(self, command):
        """Executes a command on all servers.

        :param command: Command to be executed
        :type command: string

        :return: dict of hostname -> (stdin, stdout, stderr) or "offline"
        :rtype: dict
        """
        output = {}
        for s in self.servers:
            if s.connect():
                output[s.hostname] = s.execute_command(command)
            else:
                output[s.hostname] = "offline"
            s.disconnect()
        self.log.append(
            str(datetime.datetime.now())
            + " Broadcast "
            + command
            + " sent to all servers"
        )
        return output

    def broadcast_kill_all(self):
        """Kills all processes for the user on all servers.

        :return: dict of hostname -> (stdin, stdout, stderr) or "offline"
        :rtype: dict
        """
        output = {}
        for s in self.servers:
            if s.connect():
                output[s.hostname] = s.kill_all_processes()
            else:
                output[s.hostname] = "offline"
            s.disconnect()
        self.log.append(
            str(datetime.datetime.now()) + " Kill all broadcast sent to all servers"
        )
        return output

    def broadcast_kill_all_screens(self):
        """Kills all screen sessions on all servers."""
        self.broadcast_command("killall -15 screen")

    def distribute_jobs(self, jobs, status=False, ignore_load=False, sort_server=True):
        """Distributes the jobs over the servers.

        :param jobs: List of SSHJobs to be executed on the servers.
        :type jobs: list[SSHJob]

        :param status: If true prints info about which job was started on which server.
        :type status: bool

        :param ignore_load: If true, starts the job without caring about the current load.
        :type ignore_load: bool

        :param sort_server: If True, servers are sorted by load (descending free CPU).
        :type sort_server: bool

        :return: (list_of_started_jobs, list_of_remaining_jobs)
        :rtype: (list, list)
        """
        self.get_servers_status()

        if sort_server:
            # Sort servers by the last known _free_cpus_last_request
            self.servers.sort(key=lambda x: x._free_cpus_last_request, reverse=True)

        # Sort jobs by num_threads descending
        jobs.sort(key=lambda x: x.num_threads, reverse=True)

        started_job = []
        # Loop over servers
        for server in self.servers:
            if status:
                print("Server:", server.hostname)
            server.connect()
            server.get_server_info()

            if ignore_load:
                num_free_cores = server.max_cpus_usage
            else:
                load_vals = server.get_server_load()
                if load_vals[0] is None:
                    num_free_cores = server.max_cpus_usage
                else:
                    free_cores = server.cpu_count - load_vals[0]
                    if free_cores > server.max_cpus_usage:
                        free_cores = server.max_cpus_usage
                    num_free_cores = free_cores

            if status:
                print("\tFree cores:", num_free_cores)
                print("\tJobs started:")

            started_job_index = []
            for j in range(len(jobs)):
                if num_free_cores < 1:
                    break
                needed_threads = jobs[j].num_threads
                if needed_threads <= num_free_cores:
                    # Start the job
                    server.execute_command(jobs[j].command)
                    self.log.append(
                        str(datetime.datetime.now())
                        + " Job "
                        + jobs[j].command
                        + " started on "
                        + server.hostname
                    )
                    if status:
                        print("\t\t", jobs[j].command)
                    started_job_index.append(jobs[j])
                    started_job.append(jobs[j])
                    num_free_cores -= needed_threads

            # Remove started jobs from the original list
            for job_st in started_job_index:
                jobs.remove(job_st)

            if status:
                print("\tNow Free cores:", num_free_cores)
            server.disconnect()
        return started_job, jobs

    def get_servers_status(self):
        """Reads the status of all servers and returns it as a list. Also prints if needed.

        :return: (header, list_of_info)
        :rtype: (list[str], list[list[any]])
        """
        results = []
        header = [
            "hostname       ",
            "status         ",
            "user processes ",
            "user screens   ",
            "sys load(%)1m  ",
            "sys load(%)5m  ",
            "sys load(%)15m ",
            "used memory(%) ",
            "free cpus 1min ",
            "free cpus 5min ",
            "free cpus 15min",
            "free memory(MB)",
        ]
        for s in self.servers:
            if s.connect():
                load = s.get_server_load()  # (load1, load5, load15, used_mem)
                processes = s.get_number_users_processes()
                screens = s.get_number_users_screens()
                if load[0] is not None and s.cpu_count > 0:
                    load1_pct = _torch_div_and_mul(load[0], s.cpu_count)
                    load5_pct = _torch_div_and_mul(load[1], s.cpu_count)
                    load15_pct = _torch_div_and_mul(load[2], s.cpu_count)
                    used_mem_pct = _torch_div_and_mul(load[3], s.memory_size)
                    free_cp1 = _torch_sub(s.cpu_count, load[0])
                    free_cp5 = _torch_sub(s.cpu_count, load[1])
                    free_cp15 = _torch_sub(s.cpu_count, load[2])
                    free_mem = s.memory_size - load[3]
                    results.append(
                        [
                            s.hostname,
                            "online",
                            processes,
                            screens,
                            load1_pct,
                            load5_pct,
                            load15_pct,
                            used_mem_pct,
                            free_cp1,
                            free_cp5,
                            free_cp15,
                            free_mem,
                        ]
                    )
                else:
                    # Could not retrieve load
                    results.append(
                        [
                            s.hostname,
                            "online",
                            processes,
                            screens,
                            "-",
                            "-",
                            "-",
                            "-",
                            "-",
                            "-",
                            "-",
                            "-",
                        ]
                    )
            else:
                results.append(
                    [
                        s.hostname,
                        "offline",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                    ]
                )
            s.disconnect()

        # Sort by the "free cpus 5min" index => 9, descending if numeric
        def sort_key(row):
            val = row[9]
            if isinstance(val, float):
                return val
            return -999999  # if it's '-', treat as negative

        results.sort(key=sort_key, reverse=True)

        # Print the table
        for h in header:
            print(h, end="\t")
        print("")
        for r in results:
            for col in r:
                if isinstance(col, float):
                    print(f"{col:.2f}".ljust(8), end="\t")
                else:
                    print(str(col).ljust(8), end="\t")
            print("")
        return header, results

    def get_servers_info(self, status=True):
        """Reads the status of all servers, storing in the SSHConnection objects.
           Additionally print to the console if status == True.

        :param status: If true prints info.
        :type status: bool
        """
        if status:
            print(
                "Hostname\tStatus\tCPU count\tMax CPU usage\tMemory size\tCPU speed\tCPU architecture"
            )
        for s in self.servers:
            onoff = "offline"
            if s.connect():
                onoff = s.get_server_info()
                s.disconnect()
            if status:
                print(
                    s.hostname,
                    "\t",
                    onoff,
                    "\t",
                    s.cpu_count,
                    "\t\t",
                    s.max_cpus_usage,
                    "\t\t",
                    s.memory_size,
                    "\t\t",
                    s.cpu_speed,
                    "\t\t",
                    s.architecture,
                )
