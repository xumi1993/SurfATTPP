#include "h5io.h"

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

H5IO::H5IO(const std::string &path, Mode mode) {
    switch (mode) {
    case RDONLY:
        file_     = H5::H5File(path, H5F_ACC_RDONLY);
        readonly_ = true;
        break;
    case RDWR:
        file_ = H5::H5File(path, H5F_ACC_RDWR);
        break;
    case TRUNC:
        file_ = H5::H5File(path, H5F_ACC_TRUNC);
        break;
    }
}

// ---------------------------------------------------------------------------
// String attribute
// ---------------------------------------------------------------------------

void H5IO::write_attr(const std::string &obj_path,
                      const std::string &attr_name,
                      const std::string &value) {
    ensure_not_readonly();
    H5::StrType   str_type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::DataSpace scalar(H5S_SCALAR);

    H5::H5Object *obj = nullptr;
    H5::Group     grp;
    H5::DataSet   ds;

    if (obj_path == "/" || file_.nameExists(obj_path) == false ||
        file_.childObjType(obj_path) == H5O_TYPE_GROUP) {
        grp = (obj_path == "/") ? file_.openGroup("/")
                                : file_.openGroup(obj_path);
        obj = &grp;
    } else {
        ds  = file_.openDataSet(obj_path);
        obj = &ds;
    }

    if (obj->attrExists(attr_name))
        obj->removeAttr(attr_name);

    H5::Attribute attr = obj->createAttribute(attr_name, str_type, scalar);
    const char   *cstr = value.c_str();
    attr.write(str_type, &cstr);
}

std::string H5IO::read_attr(const std::string &obj_path,
                             const std::string &attr_name) const {
    H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);

    H5::H5Object *obj = nullptr;
    H5::Group     grp;
    H5::DataSet   ds;

    if (obj_path == "/" || file_.childObjType(obj_path) == H5O_TYPE_GROUP) {
        grp = (obj_path == "/") ? file_.openGroup("/")
                                : file_.openGroup(obj_path);
        obj = &grp;
    } else {
        ds  = file_.openDataSet(obj_path);
        obj = &ds;
    }

    H5::Attribute attr = obj->openAttribute(attr_name);
    std::string   val;
    char         *cstr = nullptr;
    attr.read(str_type, &cstr);
    if (cstr) { val = cstr; free(cstr); }
    return val;
}

// ---------------------------------------------------------------------------
// Existence check
// ---------------------------------------------------------------------------

bool H5IO::exists(const std::string &name) const {
    return file_.nameExists(name);
}
