"""
Pydantic request/response schemas for the Track 1 API.
"""

from pydantic import BaseModel, Field


class PipelineRequest(BaseModel):
    """Request body for POST /api/pipeline/run."""
    video_path: str = Field(
        ...,
        description="Absolute path to the input .mp4 video file.",
        examples=["path/to/videos/intersection_01.mp4"],
    )
    camera_id: str | None = Field(
        None,
        description="Optional ID of the camera to associate with this video.",
    )
    src_pts: list[list[int]] | None = Field(
        None,
        description="Optional 4x2 matrix (list of 4 [x,y] points) for homography calibration. Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]].",
    )


class FeedStartRequest(BaseModel):
    """Request body for POST /api/feeds/{video_id}/start."""
    camera_id: str | None = Field(
        None,
        description="Optional ID of the camera to associate with this feed.",
    )
    src_pts: list[list[int]] | None = Field(
        None,
        description="Optional 4x2 matrix (list of 4 [x,y] points) for homography calibration. Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]].",
    )



class PipelineResponse(BaseModel):
    """Immediate response after triggering the pipeline."""
    event_id: str | None = Field(
        None,
        description="Unique event identifier. Null if no event was detected.",
    )
    status: str = Field(
        ...,
        description="Current pipeline status: 'processing', 'no_event', or 'error'.",
    )
    message: str = Field(
        ...,
        description="Human-readable status message.",
    )


class EventDetail(BaseModel):
    """Full detail of a registered event (mirrors Master_Event_Log)."""
    Event_ID: str
    Trigger_Time: float
    Raw_Video_Path: str
    Causal_CSV_Path: str
    Crops_Dir_Path: str
    Duration_s: float | None = None
    Status: str
    Source_Video_Path: str | None = None
    Video_ID: str | None = None


class EventList(BaseModel):
    """Wrapper for listing events."""
    events: list[EventDetail]


class VideoSource(BaseModel):
    """A registered video source / camera feed."""
    Video_ID: str
    Label: str
    File_Path: str
    Added_At: float


class VideoSourceList(BaseModel):
    """Wrapper for listing video sources."""
    sources: list[VideoSource]
