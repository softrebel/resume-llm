from pydantic import BaseModel


class ResumeBase(BaseModel):
    title: str
    bio: str
    skills: str = ""
    experiences: str = ""
    educations: str = ""
    projects: str = ""
    courses: str = ""
    publications: str = ""


class ResumeGenerated(BaseModel):
    title: str
    bio: str
    skills: list[str] = []
    experiences: list[str] = []
    educations: list[str] = []
    projects: list[str] = []
    courses: list[str] = []
    publications: list[str] = []
