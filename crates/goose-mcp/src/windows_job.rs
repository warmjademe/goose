// Windows-specific MCP child cleanup via Job Object
// This module provides a minimal Windows API surface to prepare for
// cleaning up MCP child processes by attaching them to a Job Object
// that is terminated when the parent Goose/MCP process exits.
//
// Note: This file is Windows-only and guarded by cfg(windows).

#![allow(dead_code)]

#[cfg(windows)]
mod windows_impl {
    use std::mem::{size_of, zeroed};
    use std::ptr::null_mut;
    use std::sync::OnceLock;

    use winapi::shared::minwindef::FALSE;
    use winapi::um::handleapi::CloseHandle;
    use winapi::um::processthreadsapi::OpenProcess;
    use winapi::um::jobapi2::{AssignProcessToJobObject, CreateJobObjectW};
    use winapi::um::winbase::SetInformationJobObject;
    use winapi::um::winnt::{HANDLE, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, PROCESS_TERMINATE, PROCESS_SET_INFORMATION};

    static JOB_HANDLE: OnceLock<HANDLE> = OnceLock::new();

    pub fn ensure_job_object() -> Option<HANDLE> {
        JOB_HANDLE.get_or_try_init(|| {
            unsafe {
                // Create a new Job Object
                let job = CreateJobObjectW(null_mut(), null_mut());
                if job.is_null() {
                    return Err(std::io::Error::last_os_error());
                }

                // Enable the Kill-On-Job-Close flag so all processes in the job are terminated
                let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = zeroed();
                // BasicLimitInformation is where we set LimitFlags
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;

                // Apply the information to the job object
                // SAFETY: Pass the correct information class and a pointer to the information struct
                let _ = SetInformationJobObject(
                    job,
                    JobObjectExtendedLimitInformation,
                    &mut info as *mut _ as *mut _,
                    size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                );

                Ok(job)
            }
        }).ok().cloned()
    }

    pub fn attach_pid_to_job(pid: u32) {
        if let Some(job) = JOB_HANDLE.get() {
            unsafe {
                // Open the target process with the rights we need
                let proc = OpenProcess(PROCESS_TERMINATE | PROCESS_SET_INFORMATION, FALSE, pid);
                if !proc.is_null() {
                    let _ = AssignProcessToJobObject(*job, proc);
                    CloseHandle(proc);
                }
            }
        }
    }

    pub fn init_windows_cleanup() {
        let _ = ensure_job_object();
    }

    pub fn windows_cleanup_enabled() -> bool {
        JOB_HANDLE.get().is_some()
    }
}

#[cfg(windows)]
pub use windows_impl::{ensure_job_object, attach_pid_to_job, init_windows_cleanup, windows_cleanup_enabled};

#[cfg(not(windows))]
compile_error!("windows_job.rs is Windows-only.");
