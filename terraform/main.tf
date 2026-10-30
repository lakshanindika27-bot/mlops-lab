terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

provider "docker" {}

variable "host_port" {
  type    = number
  default = 8002
}

data "docker_image" "api" {
  name = "mlops-lab:v5"
}

resource "docker_container" "api" {
  name  = "mlops-api-tf"
  image = data.docker_image.api.id

  ports {
    internal = 8000
    external = var.host_port
  }
}

output "api_url" {
  value = "http://localhost:${var.host_port}"
}
